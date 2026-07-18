"""SQLite-backed local knowledge index (chunks + embedding blobs)."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("STDMS.Agent.RAG.Store")


def _pack_embedding(vec: Sequence[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr.tobytes()


def _unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class KnowledgeStore:
    """Local air-gap vector store under data/knowledge_index/."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL UNIQUE,
                title TEXT,
                content_hash TEXT,
                mtime REAL,
                indexed_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                meta_json TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
        conn.commit()
        conn.close()

    def count_chunks(self) -> int:
        conn = self._connect()
        n = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        conn.close()
        return n

    def count_documents(self) -> int:
        conn = self._connect()
        n = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        conn.close()
        return n

    def get_document_hash(self, source_path: str) -> Optional[str]:
        conn = self._connect()
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        conn.close()
        return str(row["content_hash"]) if row else None

    def upsert_document_chunks(
        self,
        *,
        source_path: str,
        title: str,
        content_hash: str,
        mtime: float,
        chunks: list[dict[str, Any]],
        indexed_at: str,
    ) -> int:
        """Replace chunks for a source. Each chunk: {text, embedding, meta?}."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT id FROM documents WHERE source_path = ?", (source_path,))
        row = cur.fetchone()
        if row:
            doc_id = int(row["id"])
            cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            cur.execute(
                """
                UPDATE documents
                SET title = ?, content_hash = ?, mtime = ?, indexed_at = ?
                WHERE id = ?
                """,
                (title, content_hash, mtime, indexed_at, doc_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO documents (source_path, title, content_hash, mtime, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_path, title, content_hash, mtime, indexed_at),
            )
            doc_id = int(cur.lastrowid)

        for i, ch in enumerate(chunks):
            emb = ch.get("embedding") or []
            cur.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, text, embedding, meta_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    i,
                    ch["text"],
                    _pack_embedding(emb),
                    json.dumps(ch.get("meta") or {}, ensure_ascii=False),
                ),
            )
        conn.commit()
        conn.close()
        return doc_id

    def delete_missing_sources(self, keep_paths: set[str]) -> int:
        conn = self._connect()
        cur = conn.cursor()
        rows = cur.execute("SELECT id, source_path FROM documents").fetchall()
        removed = 0
        for row in rows:
            if row["source_path"] not in keep_paths:
                cur.execute("DELETE FROM chunks WHERE document_id = ?", (row["id"],))
                cur.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                removed += 1
        conn.commit()
        conn.close()
        return removed

    def all_chunks_with_embeddings(self) -> list[dict[str, Any]]:
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT c.id, c.text, c.embedding, c.meta_json, c.chunk_index,
                   d.source_path, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
        conn.close()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            out.append(
                {
                    "id": int(row["id"]),
                    "text": row["text"],
                    "embedding": _unpack_embedding(row["embedding"]),
                    "meta": meta,
                    "chunk_index": int(row["chunk_index"]),
                    "source_path": row["source_path"],
                    "title": row["title"] or Path(row["source_path"]).stem,
                }
            )
        return out
