"""NetworkX + SQLite graph memory (air-gap default backend)."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from app.agent.memory.protocol import node_id

logger = logging.getLogger("STDMS.Agent.GraphMemory")

DEFAULT_DB = Path("data") / "agent_graph.sqlite"


def default_graph_db_path(data_dir: Optional[str] = None) -> str:
    if data_dir:
        return str(Path(data_dir) / "agent_graph.sqlite")
    return str(DEFAULT_DB)


class NetworkXGraphStore:
    """In-memory MultiDiGraph with write-through SQLite persistence."""

    def __init__(self, db_path: Optional[str] = None, *, data_dir: Optional[str] = None):
        self.db_path = db_path or default_graph_db_path(data_dir)
        self._lock = threading.RLock()
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()
        self._ensure_db()
        self._load()

    # ------------------------------------------------------------------ API

    def upsert_node(
        self,
        node_type: str,
        key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> str:
        nid = node_id(node_type, key)
        props = dict(props or {})
        props.setdefault("node_type", (node_type or "Node").strip() or "Node")
        props.setdefault("key", str(key))
        props["updated_at"] = time.time()
        with self._lock:
            if self._g.has_node(nid):
                existing = dict(self._g.nodes[nid])
                existing.update(props)
                self._g.nodes[nid].clear()
                self._g.nodes[nid].update(existing)
                store_props = dict(self._g.nodes[nid])
            else:
                self._g.add_node(nid, **props)
                store_props = dict(props)
            self._persist_node(nid, store_props)
        return nid

    def upsert_edge(
        self,
        src_type: str,
        src_key: str,
        rel: str,
        dst_type: str,
        dst_key: str,
        props: Optional[dict[str, Any]] = None,
    ) -> None:
        rel = (rel or "RELATED").strip() or "RELATED"
        src = self.upsert_node(src_type, src_key)
        dst = self.upsert_node(dst_type, dst_key)
        edge_props = dict(props or {})
        edge_props["rel"] = rel
        edge_props["updated_at"] = time.time()
        with self._lock:
            # Replace prior edges of same rel between pair (deterministic)
            to_remove = [
                k
                for k, d in self._g.get_edge_data(src, dst, default={}).items()
                if (d or {}).get("rel") == rel
            ]
            for k in to_remove:
                self._g.remove_edge(src, dst, key=k)
            self._g.add_edge(src, dst, key=rel, **edge_props)
            self._persist_edge(src, dst, rel, edge_props)

    def get_node(self, node_type: str, key: str) -> Optional[dict[str, Any]]:
        nid = node_id(node_type, key)
        with self._lock:
            if not self._g.has_node(nid):
                return None
            data = dict(self._g.nodes[nid])
            data["id"] = nid
            return data

    def neighbors(
        self,
        node_type: str,
        key: str,
        *,
        rel: Optional[str] = None,
        direction: str = "out",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        nid = node_id(node_type, key)
        limit = max(1, min(int(limit), 200))
        out: list[dict[str, Any]] = []
        with self._lock:
            if not self._g.has_node(nid):
                return []
            if direction in ("out", "both"):
                for _, dst, data in self._g.out_edges(nid, data=True):
                    if rel and data.get("rel") != rel:
                        continue
                    out.append(self._edge_hit(nid, dst, data, direction="out"))
                    if len(out) >= limit:
                        return out[:limit]
            if direction in ("in", "both"):
                for src, _, data in self._g.in_edges(nid, data=True):
                    if rel and data.get("rel") != rel:
                        continue
                    out.append(self._edge_hit(src, nid, data, direction="in"))
                    if len(out) >= limit:
                        break
        return out[:limit]

    def recall(
        self,
        query: str,
        *,
        tab_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 50))
        tokens = _tokenize(query)
        hits: list[tuple[float, dict[str, Any]]] = []

        with self._lock:
            seed_ids: list[str] = []
            if tab_id:
                tid = node_id("Tab", tab_id)
                if self._g.has_node(tid):
                    seed_ids.append(tid)
                    for _, dst in self._g.out_edges(tid):
                        seed_ids.append(dst)
                    for src, _ in self._g.in_edges(tid):
                        seed_ids.append(src)

            candidates = seed_ids or list(self._g.nodes())
            seen: set[str] = set()
            for nid in candidates:
                if nid in seen or not self._g.has_node(nid):
                    continue
                seen.add(nid)
                data = dict(self._g.nodes[nid])
                score = _score_node(data, tokens, prefer_tab=tab_id)
                if tab_id and data.get("node_type") == "Tab" and data.get("key") == tab_id:
                    score += 5.0
                if score <= 0 and not tab_id:
                    continue
                if score <= 0 and tab_id and nid not in (seed_ids[:1] if seed_ids else []):
                    # Still include 1-hop neighbors of the tab with a floor score
                    if nid in seed_ids:
                        score = 0.5
                    else:
                        continue
                title = _node_title(nid, data)
                snippet = _node_snippet(data)
                hits.append(
                    (
                        score,
                        {
                            "id": nid,
                            "title": title,
                            "snippet": snippet,
                            "source": "graph_memory",
                            "score": round(score, 3),
                            "node_type": data.get("node_type"),
                        },
                    )
                )

            # Also surface a few relevant edges as hits when query mentions relations
            if tokens:
                for u, v, data in self._g.edges(data=True):
                    rel = str(data.get("rel") or "")
                    blob = f"{u} {rel} {v}".lower()
                    if any(t in blob for t in tokens):
                        hits.append(
                            (
                                1.0 + sum(1 for t in tokens if t in blob),
                                {
                                    "id": f"edge:{u}-{rel}-{v}",
                                    "title": f"{u} -[{rel}]-> {v}",
                                    "snippet": json.dumps(
                                        {k: data[k] for k in data if k != "updated_at"},
                                        ensure_ascii=False,
                                        default=str,
                                    )[:300],
                                    "source": "graph_memory",
                                    "score": 1.0,
                                    "node_type": "Edge",
                                },
                            )
                        )

        hits.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate by id
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for _, hit in hits:
            hid = str(hit.get("id"))
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            out.append(hit)
            if len(out) >= limit:
                break
        return out

    def sync_fleet(self, host: Any) -> dict[str, Any]:
        from app.agent.memory.sync import sync_fleet_into

        return sync_fleet_into(self, host)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            types: dict[str, int] = {}
            for _, data in self._g.nodes(data=True):
                t = str(data.get("node_type") or "Unknown")
                types[t] = types.get(t, 0) + 1
            return {
                "backend": "networkx",
                "nodes": self._g.number_of_nodes(),
                "edges": self._g.number_of_edges(),
                "node_types": types,
                "db_path": self.db_path,
            }

    def clear(self) -> None:
        with self._lock:
            self._g.clear()
            conn = self._connect()
            try:
                conn.execute("DELETE FROM edges")
                conn.execute("DELETE FROM nodes")
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ persistence

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    props_json TEXT NOT NULL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    props_json TEXT NOT NULL,
                    updated_at REAL,
                    PRIMARY KEY (src_id, dst_id, rel)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _load(self) -> None:
        conn = self._connect()
        try:
            for row in conn.execute("SELECT id, props_json FROM nodes"):
                try:
                    props = json.loads(row["props_json"] or "{}")
                except Exception:
                    props = {}
                self._g.add_node(row["id"], **props)
            for row in conn.execute("SELECT src_id, dst_id, rel, props_json FROM edges"):
                try:
                    props = json.loads(row["props_json"] or "{}")
                except Exception:
                    props = {}
                props.setdefault("rel", row["rel"])
                if not self._g.has_node(row["src_id"]):
                    self._g.add_node(row["src_id"])
                if not self._g.has_node(row["dst_id"]):
                    self._g.add_node(row["dst_id"])
                self._g.add_edge(row["src_id"], row["dst_id"], key=row["rel"], **props)
        finally:
            conn.close()

    def _persist_node(self, nid: str, props: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO nodes (id, node_type, key, props_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type=excluded.node_type,
                    key=excluded.key,
                    props_json=excluded.props_json,
                    updated_at=excluded.updated_at
                """,
                (
                    nid,
                    str(props.get("node_type") or "Node"),
                    str(props.get("key") or ""),
                    json.dumps(props, ensure_ascii=False, default=str),
                    float(props.get("updated_at") or time.time()),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_edge(
        self, src: str, dst: str, rel: str, props: dict[str, Any]
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO edges (src_id, dst_id, rel, props_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(src_id, dst_id, rel) DO UPDATE SET
                    props_json=excluded.props_json,
                    updated_at=excluded.updated_at
                """,
                (
                    src,
                    dst,
                    rel,
                    json.dumps(props, ensure_ascii=False, default=str),
                    float(props.get("updated_at") or time.time()),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _edge_hit(
        src: str, dst: str, data: dict[str, Any], *, direction: str
    ) -> dict[str, Any]:
        return {
            "id": dst if direction == "out" else src,
            "rel": data.get("rel"),
            "direction": direction,
            "props": {k: v for k, v in data.items() if k not in ("updated_at",)},
        }


def _tokenize(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]{2,}", (query or "").lower()) if t]


def _score_node(data: dict[str, Any], tokens: list[str], *, prefer_tab: Optional[str]) -> float:
    if not tokens:
        # Without a query, prefer unhealthy / pending operational nodes
        score = 0.1
        health = str(data.get("health") or data.get("health_state") or "").lower()
        if health in ("warning", "critical", "error", "anomaly"):
            score += 2.0
        if data.get("node_type") in ("Draft", "RetrainSignal"):
            score += 1.5
        if prefer_tab and data.get("key") == prefer_tab:
            score += 3.0
        return score
    blob = " ".join(
        str(v) for k, v in data.items() if k not in ("updated_at",) and v is not None
    ).lower()
    blob = f"{data.get('node_type', '')} {data.get('key', '')} {blob}".lower()
    return float(sum(1.5 if t in blob else 0.0 for t in tokens))


def _node_title(nid: str, data: dict[str, Any]) -> str:
    t = data.get("node_type") or "Node"
    if t == "Tab":
        return f"Tab {data.get('title') or data.get('key') or nid}"
    if t == "Decision":
        return f"Decision #{data.get('key')} ({data.get('outcome') or 'n/a'})"
    if t == "Draft":
        return f"Draft {data.get('kind') or data.get('key')} [{data.get('status') or 'pending'}]"
    if t == "RetrainSignal":
        return f"RetrainSignal {data.get('key')} drift={data.get('drift_score')}"
    if t == "Mode":
        return f"Mode {data.get('key')} (×{data.get('threshold_scale', 1)})"
    if t == "Model":
        return f"Model {data.get('key')} ({data.get('model_type') or 'n/a'})"
    return nid


def _node_snippet(data: dict[str, Any]) -> str:
    skip = {"updated_at", "node_type", "key"}
    parts = []
    for k, v in data.items():
        if k in skip or v is None or v == "":
            continue
        parts.append(f"{k}={v}")
        if len(parts) >= 8:
            break
    return "; ".join(parts)[:400]
