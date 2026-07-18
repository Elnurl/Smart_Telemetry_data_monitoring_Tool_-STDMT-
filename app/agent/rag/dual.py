"""Dual space/ground knowledge stores (SQLite embeddings — Chroma-compatible API shape)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from app.agent.ollama_client import DEFAULT_OLLAMA_URL
from app.agent.rag.retrieve import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_PATH,
    knowledge_status,
    search_knowledge_index,
)

logger = logging.getLogger("STDMS.Agent.RAG.Dual")

Segment = Literal["space", "ground", "both", "auto"]

DEFAULT_DOCS_DIR = Path("data") / "knowledge"
SPACE_INDEX_PATH = Path("data") / "knowledge_index" / "rag_space.sqlite"
GROUND_INDEX_PATH = Path("data") / "knowledge_index" / "rag_ground.sqlite"

_SPACE_KEYWORDS = (
    "satellite",
    "peyk",
    "telemetry",
    "telemetriya",
    "sensor",
    "battery",
    "eclipse",
    "maneuver",
    "manevr",
    "fsm",
    "mission mode",
    "threshold",
    "thermal",
    "tcs",
    "adcs",
    "power",
    "eps",
    "cdh",
    "solar",
    "orbit",
    "safe_mode",
    "safe mode",
    "hardware",
    "payload",
    "bus voltage",
)

_GROUND_KEYWORDS = (
    "alert",
    "maintenance",
    "prosedur",
    "procedure",
    "operator",
    "report",
    "sop",
    "ground",
    "yer stansiya",
    "approve",
    "draft",
    "on-call",
    "on call",
    "escalate",
    "acknowledge",
    "retrain signal",
    "pending",
    "dashboard",
)

_BOTH_KEYWORDS = (
    "anomaly",
    "anomaliya",
    "health",
    "monitoring",
    "drift",
    "warning",
    "critical",
    "obs",
)


def classify_segment(query: str) -> Segment:
    """Fast keyword classification — no LLM call."""
    q = (query or "").strip().lower()
    if not q:
        return "both"
    space_hits = sum(1 for k in _SPACE_KEYWORDS if k in q)
    ground_hits = sum(1 for k in _GROUND_KEYWORDS if k in q)
    both_hits = sum(1 for k in _BOTH_KEYWORDS if k in q)

    if both_hits and space_hits == ground_hits:
        return "both"
    if space_hits > ground_hits and space_hits > 0:
        return "space"
    if ground_hits > space_hits and ground_hits > 0:
        return "ground"
    if both_hits > 0:
        return "both"
    # Default: procedures/SOP-ish → ground; tech → space; else both
    if re.search(r"\b(how to|what is the (alert|sop|procedure))\b", q):
        return "ground"
    if re.search(r"\b(threshold|eclipse|battery|sensor|mode)\b", q):
        return "space"
    return "both"


def dual_index_paths() -> dict[str, Path]:
    return {"space": SPACE_INDEX_PATH, "ground": GROUND_INDEX_PATH}


def dual_knowledge_status(
    *,
    space_index: str | Path = SPACE_INDEX_PATH,
    ground_index: str | Path = GROUND_INDEX_PATH,
    legacy_index: str | Path = DEFAULT_INDEX_PATH,
) -> dict[str, Any]:
    space = knowledge_status(space_index)
    ground = knowledge_status(ground_index)
    legacy = knowledge_status(legacy_index)
    total_chunks = int(space.get("chunks") or 0) + int(ground.get("chunks") or 0)
    if total_chunks > 0:
        status = "available"
    elif int(legacy.get("chunks") or 0) > 0:
        status = "available"
        total_chunks = int(legacy.get("chunks") or 0)
    else:
        status = "empty"
    return {
        "status": status,
        "space": space,
        "ground": ground,
        "legacy": legacy,
        "documents": int(space.get("documents") or 0) + int(ground.get("documents") or 0),
        "chunks": total_chunks,
        "mode": "dual_space_ground",
    }


class DualStoreRetriever:
    """Two SQLite embedding stores: stdms_space + stdms_ground."""

    def __init__(
        self,
        *,
        space_index: str | Path = SPACE_INDEX_PATH,
        ground_index: str | Path = GROUND_INDEX_PATH,
        legacy_index: str | Path = DEFAULT_INDEX_PATH,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_fn: Optional[Callable[..., Optional[list[float]]]] = None,
    ):
        self.space_index = Path(space_index)
        self.ground_index = Path(ground_index)
        self.legacy_index = Path(legacy_index)
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.embed_fn = embed_fn

    def _classify(self, query: str) -> Segment:
        return classify_segment(query)

    def retrieve(
        self,
        query: str,
        segment: str = "auto",
        top_k: int = 3,
        *,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        seg: Segment
        if segment in ("space", "ground", "both", "auto"):
            seg = segment  # type: ignore[assignment]
        else:
            seg = "auto"
        if seg == "auto":
            seg = self._classify(query)

        if seg == "space":
            return self._query_store(query, self.space_index, "space", top_k, min_score)
        if seg == "ground":
            return self._query_store(query, self.ground_index, "ground", top_k, min_score)

        # both — or fallback when a side is empty
        per = max(1, int(top_k) // 2) if int(top_k) > 1 else 1
        space = self._query_store(query, self.space_index, "space", per, min_score)
        ground = self._query_store(query, self.ground_index, "ground", per, min_score)
        merged = space + ground
        if not merged:
            # Legacy single-store fallback (pre-Faza-4 indexes)
            legacy = self._query_store(query, self.legacy_index, "legacy", top_k, min_score)
            return legacy
        merged.sort(key=lambda h: float(h.get("score") or 0), reverse=True)
        return merged[: max(1, int(top_k))]

    def search(
        self,
        query: str,
        *,
        segment: str = "auto",
        top_k: int = 6,
        min_score: float = 0.25,
    ) -> dict[str, Any]:
        resolved = self._classify(query) if segment == "auto" else segment
        hits = self.retrieve(query, segment=segment, top_k=top_k, min_score=min_score)
        status = dual_knowledge_status(
            space_index=self.space_index,
            ground_index=self.ground_index,
            legacy_index=self.legacy_index,
        )
        return {
            "hits": hits,
            "knowledge_status": status.get("status") if hits else (
                status.get("status") if status.get("chunks") else "no_match"
            ),
            "query": (query or "").strip(),
            "segment": resolved,
            "note": (
                f"Retrieved {len(hits)} chunk(s) from dual store (segment={resolved})."
                if hits
                else "No matching chunk in space/ground indexes. Rebuild Knowledge."
            ),
            "index": status,
            "mode": "dual_space_ground",
        }

    def _query_store(
        self,
        query: str,
        index_path: Path,
        segment_label: str,
        top_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        if not index_path.is_file():
            return []
        out = search_knowledge_index(
            query,
            top_k=top_k,
            min_score=min_score,
            index_path=index_path,
            ollama_url=self.ollama_url,
            embed_model=self.embed_model,
            embed_fn=self.embed_fn,
        )
        hits: list[dict[str, Any]] = []
        for hit in out.get("hits") or []:
            item = dict(hit)
            source = str(item.get("source") or "")
            # Normalize to space/... or ground/... for routing assertions
            if segment_label in ("space", "ground"):
                if not (source.startswith(f"{segment_label}/") or f"/{segment_label}/" in source.replace("\\", "/")):
                    # Prefer basename under segment prefix
                    name = Path(source).name if source else "unknown.md"
                    source = f"{segment_label}/{name}"
                    item["source"] = source
            item["segment"] = segment_label
            hits.append(item)
        return hits
