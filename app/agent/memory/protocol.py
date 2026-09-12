"""Backend-agnostic graph memory contract (NetworkX now, Neo4j later)."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


def node_id(node_type: str, key: str) -> str:
    """Canonical node id: ``{Type}:{key}``."""
    t = (node_type or "Node").strip() or "Node"
    k = str(key or "").strip() or "_"
    return f"{t}:{k}"


@runtime_checkable
class GraphMemoryStore(Protocol):
    """Swappable graph memory backend for agent operational recall."""

    def upsert_node(
        self,
        node_type: str,
        key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create or update a typed node; return node id."""
        ...

    def upsert_edge(
        self,
        src_type: str,
        src_key: str,
        rel: str,
        dst_type: str,
        dst_key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create or update a directed relationship between two nodes."""
        ...

    def get_node(self, node_type: str, key: str) -> Optional[dict[str, Any]]:
        ...

    def neighbors(
        self,
        node_type: str,
        key: str,
        *,
        rel: Optional[str] = None,
        direction: str = "out",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        ...

    def recall(
        self,
        query: str,
        *,
        tab_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return RAG-shaped hits: ``title``, ``snippet``, ``source``, ``score``."""
        ...

    def sync_fleet(self, host: Any) -> dict[str, Any]:
        """Pull live fleet / drafts / FSM / audit into the graph."""
        ...

    def stats(self) -> dict[str, Any]:
        ...

    def clear(self) -> None:
        ...
