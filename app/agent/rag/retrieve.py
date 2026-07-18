"""Retrieve top-k chunks by hybrid score (vector + keyword) for all documents."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from app.agent.ollama_client import DEFAULT_OLLAMA_URL, embed_ollama
from app.agent.rag.store import KnowledgeStore

logger = logging.getLogger("STDMS.Agent.RAG.Retrieve")

DEFAULT_INDEX_PATH = Path("data") / "knowledge_index" / "rag.sqlite"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

_STOPWORDS = frozenset(
    {
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "about",
        "explain",
        "tell",
        "me",
        "please",
        "nə",
        "ne",
        "dir",
        "dır",
        "haqqında",
        "nedir",
        "nədir",
        "olan",
        "nədir",
    }
)


def cosine_scores(query_vec: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if matrix.size == 0 or q.size == 0:
        return np.asarray([], dtype=np.float32)
    dim = min(q.shape[0], matrix.shape[1])
    q = q[:dim]
    m = matrix[:, :dim]
    qn = np.linalg.norm(q) + 1e-9
    mn = np.linalg.norm(m, axis=1) + 1e-9
    return (m @ q) / (mn * qn)


def _query_terms(query: str) -> list[str]:
    raw = (query or "").lower()
    tokens = re.findall(r"[a-zA-Z0-9ğüşıöçəĞÜŞİÖÇƏ_]{2,}", raw, flags=re.IGNORECASE)
    terms: list[str] = []
    for tok in tokens:
        t = tok.lower()
        if t in _STOPWORDS:
            continue
        if t not in terms:
            terms.append(t)
    return terms


def keyword_boost(query: str, text: str, *, title: str = "", source: str = "") -> float:
    """Boost chunks that contain exact query terms (all documents, not one name)."""
    terms = _query_terms(query)
    if not terms:
        return 0.0
    hay = (text or "").lower()
    title_hay = f"{title} {Path(source).stem}".lower()
    boost = 0.0
    matched = 0
    for term in terms:
        in_body = term in hay
        in_title = term in title_hay
        if in_body or in_title:
            matched += 1
            boost += 0.55
            if in_body and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", hay, flags=re.IGNORECASE):
                boost += 0.35
            if in_title:
                boost += 0.45
    phrase = " ".join(terms)
    if len(phrase) >= 3 and phrase in hay:
        boost += 0.6
    # Prefer chunks covering more of the query terms
    if terms:
        boost += 0.25 * (matched / max(1, len(terms)))
    return float(min(boost, 2.5))


def focused_snippet(text: str, query: str, *, max_len: int = 700) -> str:
    """Return a window centered on the first matched query term."""
    raw = text or ""
    if len(raw) <= max_len:
        return raw
    terms = _query_terms(query)
    lower = raw.lower()
    pos = -1
    for term in terms:
        pos = lower.find(term)
        if pos >= 0:
            break
    if pos < 0:
        return raw[: max_len - 1] + "…"
    half = max_len // 2
    start = max(0, pos - half // 2)
    end = min(len(raw), start + max_len)
    start = max(0, end - max_len)
    snippet = raw[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(raw):
        snippet = snippet + "…"
    return snippet


def knowledge_status(index_path: str | Path = DEFAULT_INDEX_PATH) -> dict[str, Any]:
    path = Path(index_path)
    if not path.is_file():
        return {
            "status": "unavailable",
            "documents": 0,
            "chunks": 0,
            "index_path": str(path),
        }
    try:
        store = KnowledgeStore(path)
        docs = store.count_documents()
        chunks = store.count_chunks()
        if chunks <= 0:
            return {
                "status": "empty",
                "documents": docs,
                "chunks": 0,
                "index_path": str(path),
            }
        return {
            "status": "available",
            "documents": docs,
            "chunks": chunks,
            "index_path": str(path),
        }
    except Exception as exc:
        logger.info("knowledge_status failed: %s", exc)
        return {
            "status": "error",
            "documents": 0,
            "chunks": 0,
            "error": str(exc),
            "index_path": str(path),
        }


def search_knowledge_index(
    query: str,
    *,
    top_k: int = 6,
    min_score: float = 0.25,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_fn: Optional[Callable[..., Optional[list[float]]]] = None,
) -> dict[str, Any]:
    """Hybrid retrieve for all docs: keyword precision + vector recall."""
    q = (query or "").strip()
    status = knowledge_status(index_path)
    if not q:
        return {
            "hits": [],
            "knowledge_status": status.get("status"),
            "query": "",
            "note": "Empty query.",
        }
    if status.get("status") != "available":
        return {
            "hits": [],
            "knowledge_status": status.get("status"),
            "query": q,
            "note": "Knowledge index empty or missing. Place docs in data/knowledge and rebuild index.",
            "index": status,
        }

    store = KnowledgeStore(index_path)
    rows = store.all_chunks_with_embeddings()
    if not rows:
        return {
            "hits": [],
            "knowledge_status": "empty",
            "query": q,
            "note": "No chunks in index.",
        }

    kw_scores = np.asarray(
        [
            keyword_boost(q, r["text"], title=str(r.get("title") or ""), source=str(r.get("source_path") or ""))
            for r in rows
        ],
        dtype=np.float32,
    )

    embed = embed_fn or embed_ollama
    qvec = embed(q, base_url=ollama_url, model=embed_model)
    if qvec:
        matrix = np.stack([r["embedding"] for r in rows], axis=0)
        vec_scores = cosine_scores(qvec, matrix)
    else:
        vec_scores = np.zeros(len(rows), dtype=np.float32)
        if float(kw_scores.max(initial=0.0)) <= 0:
            return {
                "hits": [],
                "knowledge_status": "embed_unavailable",
                "query": q,
                "note": (
                    f"Could not embed query via Ollama ({embed_model}) and no keyword matches."
                ),
                "index": status,
            }

    final = vec_scores + kw_scores
    has_strong_keyword = bool(float(kw_scores.max(initial=0.0)) >= 0.5)

    # When exact terms exist in the corpus, prefer those chunks and drop unrelated vector noise.
    candidate_idx = list(range(len(rows)))
    if has_strong_keyword:
        candidate_idx = [i for i in candidate_idx if float(kw_scores[i]) >= 0.5]
        if not candidate_idx:
            candidate_idx = list(range(len(rows)))

    candidate_idx.sort(key=lambda i: float(final[i]), reverse=True)

    hits: list[dict[str, Any]] = []
    seen_sources: dict[str, int] = {}
    for idx in candidate_idx:
        score = float(final[int(idx)])
        kw = float(kw_scores[int(idx)])
        if score < min_score and kw < 0.5:
            continue
        if kw >= 0.5:
            score = max(score, 0.85)

        row = rows[int(idx)]
        source = str(row.get("source_path") or "")
        # Keep answer focused: at most 2 chunks per source unless we still need fill
        if seen_sources.get(source, 0) >= 2 and len(hits) >= max(2, int(top_k) // 2):
            continue

        text = row["text"]
        snippet = focused_snippet(text, q, max_len=750)
        hits.append(
            {
                "id": f"rag-{row['id']}",
                "title": row["title"],
                "source": source,
                "snippet": snippet,
                "score": round(score, 4),
                "vector_score": round(float(vec_scores[int(idx)]), 4),
                "keyword_boost": round(kw, 4),
                "chunk_index": row["chunk_index"],
            }
        )
        seen_sources[source] = seen_sources.get(source, 0) + 1
        if len(hits) >= max(1, int(top_k)):
            break

    return {
        "hits": hits,
        "knowledge_status": "available" if hits else "no_match",
        "query": q,
        "note": (
            f"Retrieved {len(hits)} chunk(s) via hybrid search over local documents."
            if hits
            else "No matching chunk in local knowledge. Add/rebuild documents or rephrase the question."
        ),
        "index": status,
        "mode": "hybrid_keyword_vector",
    }
