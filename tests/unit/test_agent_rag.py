"""Phase 4 RAG unit tests (mocked embeddings — no live Ollama)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.agent.rag.chunk import chunk_text
from app.agent.rag.ingest import ingest_knowledge_dir
from app.agent.rag.retrieve import cosine_scores, knowledge_status, search_knowledge_index
from app.agent.rag.store import KnowledgeStore
from app.agent.tools import search_knowledge


def _fake_embed(text: str, **kwargs):
    """Deterministic tiny embedding from character stats (tests only)."""
    t = (text or "").lower()
    return [
        float(len(t) % 97) / 97.0,
        float(t.count("anomaly")) + 0.1,
        float(t.count("retrain")) + 0.1,
        float(t.count("drift")) + 0.1,
        float(t.count("monitor")) + 0.1,
        float(sum(ord(c) for c in t[:40]) % 100) / 100.0,
    ]


def test_chunk_text_overlap():
    text = ("Paragraph one about anomalies. " * 20) + "\n\n" + ("Paragraph two about drift. " * 20)
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(chunks)


def test_ingest_skips_archive_and_offtopic_names(tmp_path: Path):
    docs = tmp_path / "knowledge"
    (docs / "_archive").mkdir(parents=True)
    (docs / "SOP-7.md").write_text("Mission SOP anomaly response procedure.", encoding="utf-8")
    (docs / "_archive" / "notes.md").write_text("Personal notes should not index.", encoding="utf-8")
    (docs / "Magoosh+IELTS+Vocabulary.pdf").write_text("%PDF-fake", encoding="utf-8")
    index = tmp_path / "rag.sqlite"
    result = ingest_knowledge_dir(docs, index_path=index, embed_fn=_fake_embed, force=True)
    assert result["ok"] is True
    assert result["documents"] == 1
    assert result["files_seen"] == 1


def test_store_and_retrieve(tmp_path: Path):
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "sop.md").write_text(
        "SOP-7 anomaly response: if drift detected, propose retrain and draft alert for approval.",
        encoding="utf-8",
    )
    index = tmp_path / "rag.sqlite"
    result = ingest_knowledge_dir(
        docs,
        index_path=index,
        embed_fn=_fake_embed,
        force=True,
    )
    assert result["ok"] is True
    assert result["chunks"] >= 1
    status = knowledge_status(index)
    assert status["status"] == "available"

    out = search_knowledge_index(
        "anomaly drift retrain procedure",
        index_path=index,
        embed_fn=_fake_embed,
        min_score=0.0,
    )
    assert out["hits"]
    assert "anomaly" in out["hits"][0]["snippet"].lower() or "SOP" in out["hits"][0]["title"]


def test_hybrid_keyword_boost_finds_rare_term(tmp_path: Path):
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "noise.md").write_text(
        "Particle detector readout analysis methods and nuclear radiation physics. " * 40,
        encoding="utf-8",
    )
    (docs / "profile.md").write_text(
        "Elnur Ahmadzade projects. Rubai Tetbiqi: Flutter esasli goal-tracking mobil tetbiq.",
        encoding="utf-8",
    )
    index = tmp_path / "rag.sqlite"
    ingest_knowledge_dir(docs, index_path=index, embed_fn=_fake_embed, force=True)
    out = search_knowledge_index("what is Rubai?", index_path=index, embed_fn=_fake_embed, min_score=0.1)
    assert out["hits"]
    joined = " ".join(h["snippet"] for h in out["hits"]).lower()
    assert "rubai" in joined
    assert out["hits"][0].get("keyword_boost", 0) > 0


def test_cosine_scores_identical():
    v = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    m = np.stack([v, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)])
    scores = cosine_scores(v, m)
    assert scores[0] > 0.99
    assert scores[1] < 0.1


def test_search_knowledge_merges_builtin(tmp_path: Path, monkeypatch):
    # Point RAG at empty index → builtin how-to still works
    empty = tmp_path / "empty.sqlite"
    KnowledgeStore(empty)  # create empty schema
    monkeypatch.setattr(
        "app.agent.rag.retrieve.DEFAULT_INDEX_PATH",
        empty,
    )
    out = search_knowledge("how to start monitoring")
    assert out["hits"]
    assert any(h.get("source") == "builtin_product_guide" for h in out["hits"])
