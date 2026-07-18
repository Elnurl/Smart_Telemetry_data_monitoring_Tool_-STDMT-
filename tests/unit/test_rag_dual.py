"""Phase 4 — dual space/ground RAG routing + FSM export."""

from __future__ import annotations

from pathlib import Path

from app.agent.rag.dual import DualStoreRetriever, classify_segment
from app.agent.rag.ingest import export_fsm_to_knowledge, ingest_dual_knowledge


def _fake_embed(text: str, **kwargs):
    t = (text or "").lower()
    return [
        float(len(t) % 97) / 97.0,
        float(t.count("eclipse")) + 0.1,
        float(t.count("threshold")) + 0.1,
        float(t.count("alert")) + 0.1,
        float(t.count("procedure")) + 0.1,
        float(t.count("anomaly")) + 0.1,
        float(sum(ord(c) for c in t[:40]) % 100) / 100.0,
    ]


def test_classify_segment_keywords():
    assert classify_segment("eclipse threshold scale") == "space"
    assert classify_segment("what is the alert procedure?") == "ground"
    assert classify_segment("why is the tab showing anomalies?") == "both"


def test_dual_store_routing(tmp_path: Path):
    docs = tmp_path / "knowledge"
    space = docs / "space"
    ground = docs / "ground"
    space.mkdir(parents=True)
    ground.mkdir(parents=True)
    (space / "eclipse_modes.md").write_text(
        "Eclipse threshold_scale is 1.5. TCS WARNING is mode-normal in eclipse.",
        encoding="utf-8",
    )
    (ground / "alert_sop.md").write_text(
        "Alert procedure: acknowledge, isolate channel, notify on-call operator.",
        encoding="utf-8",
    )

    space_idx = tmp_path / "rag_space.sqlite"
    ground_idx = tmp_path / "rag_ground.sqlite"
    legacy_idx = tmp_path / "rag_legacy.sqlite"

    # Patch dual paths via DualStoreRetriever args + ingest with explicit prefixes
    from app.agent.rag.ingest import ingest_knowledge_dir

    ingest_knowledge_dir(
        space,
        index_path=space_idx,
        embed_fn=_fake_embed,
        force=True,
        source_prefix="space",
    )
    ingest_knowledge_dir(
        ground,
        index_path=ground_idx,
        embed_fn=_fake_embed,
        force=True,
        source_prefix="ground",
    )

    r = DualStoreRetriever(
        space_index=space_idx,
        ground_index=ground_idx,
        legacy_index=legacy_idx,
        embed_fn=_fake_embed,
    )
    space_hits = r.retrieve("eclipse threshold", segment="space", top_k=3)
    assert space_hits
    assert all(str(h["source"]).startswith("space/") for h in space_hits)

    ground_hits = r.retrieve("alert procedure", segment="ground", top_k=3)
    assert ground_hits
    assert all(str(h["source"]).startswith("ground/") for h in ground_hits)

    both = r.retrieve("anomaly monitoring health", segment="both", top_k=4)
    assert both
    segs = {h.get("segment") for h in both}
    assert segs & {"space", "ground", "legacy"}


def test_fsm_export(tmp_path: Path):
    out = tmp_path / "space"
    result = export_fsm_to_knowledge(
        out,
        tabs_config={
            "tab-eclipse": {
                "title": "Eclipse monitoring",
                "current_mission_mode": "eclipse",
                "mission_modes": [
                    {"name": "nominal", "threshold_scale": 1.0, "description": "Normal ops"},
                    {
                        "name": "eclipse",
                        "threshold_scale": 1.5,
                        "description": "Eclipse thermal swings expected",
                    },
                ],
            }
        },
    )
    assert result["ok"] is True
    path = Path(result["path"])
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Eclipse monitoring" in text
    assert "threshold_scale: 1.5" in text
    assert "mode-normal" in text.lower()

    # Rebuild dual indexes with exported file present
    docs = tmp_path / "knowledge"
    (docs / "space").mkdir(parents=True)
    (docs / "ground").mkdir(parents=True)
    (docs / "space" / "mission_modes_auto.md").write_text(text, encoding="utf-8")
    (docs / "ground" / "stub.md").write_text("Ground alert procedure stub.", encoding="utf-8")

    # Point dual ingest indexes into tmp by monkeypatching paths used inside ingest_dual
    import app.agent.rag.dual as dual_mod
    import app.agent.rag.ingest as ingest_mod

    space_idx = tmp_path / "idx_space.sqlite"
    ground_idx = tmp_path / "idx_ground.sqlite"
    legacy_idx = tmp_path / "idx_legacy.sqlite"
    original_space = dual_mod.SPACE_INDEX_PATH
    original_ground = dual_mod.GROUND_INDEX_PATH
    original_default = ingest_mod.DEFAULT_INDEX_PATH
    try:
        dual_mod.SPACE_INDEX_PATH = space_idx
        dual_mod.GROUND_INDEX_PATH = ground_idx
        ingest_mod.DEFAULT_INDEX_PATH = legacy_idx
        dual = ingest_dual_knowledge(
            docs,
            embed_fn=_fake_embed,
            force=True,
            export_fsm=False,
        )
        assert dual["ok"] is True
        assert dual["space"]["documents"] >= 1

        retriever = DualStoreRetriever(
            space_index=space_idx,
            ground_index=ground_idx,
            legacy_index=legacy_idx,
            embed_fn=_fake_embed,
        )
        hits = retriever.retrieve("eclipse mode threshold", segment="space", top_k=3)
        assert hits
        joined = " ".join(h["snippet"] for h in hits).lower()
        assert "eclipse" in joined or "threshold" in joined
    finally:
        dual_mod.SPACE_INDEX_PATH = original_space
        dual_mod.GROUND_INDEX_PATH = original_ground
        ingest_mod.DEFAULT_INDEX_PATH = original_default
