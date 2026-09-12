"""Neo4j adapter seam — not implemented in Sprint 2 (air-gap / no Docker).

Callers should use ``GraphMemoryStore`` via ``get_graph_memory()``. When a Neo4j
deployment is available, implement this class against the same protocol and select
it with ``STDMS_GRAPH_BACKEND=neo4j`` + connection env vars.
"""

from __future__ import annotations

from typing import Any, Optional


_MSG = (
    "Neo4jGraphStore is a Sprint-2 seam only. "
    "Use NetworkXGraphStore (default) or implement this adapter with the neo4j "
    "Python driver and STDMS_NEO4J_URI / STDMS_NEO4J_USER / STDMS_NEO4J_PASSWORD."
)


class Neo4jGraphStore:
    """Placeholder implementing GraphMemoryStore; all methods raise."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(_MSG)

    def upsert_node(
        self,
        node_type: str,
        key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError(_MSG)

    def upsert_edge(
        self,
        src_type: str,
        src_key: str,
        rel: str,
        dst_type: str,
        dst_key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError(_MSG)

    def get_node(self, node_type: str, key: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError(_MSG)

    def neighbors(
        self,
        node_type: str,
        key: str,
        *,
        rel: Optional[str] = None,
        direction: str = "out",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(_MSG)

    def recall(
        self,
        query: str,
        *,
        tab_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(_MSG)

    def sync_fleet(self, host: Any) -> dict[str, Any]:
        raise NotImplementedError(_MSG)

    def stats(self) -> dict[str, Any]:
        raise NotImplementedError(_MSG)

    def clear(self) -> None:
        raise NotImplementedError(_MSG)
