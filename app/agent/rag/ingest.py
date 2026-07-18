"""Ingest Markdown/PDF/TXT from data/knowledge into the local RAG index."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from app.agent.ollama_client import DEFAULT_OLLAMA_URL, embed_ollama, ollama_reachable
from app.agent.rag.chunk import chunk_text
from app.agent.rag.retrieve import DEFAULT_EMBED_MODEL, DEFAULT_INDEX_PATH
from app.agent.rag.store import KnowledgeStore

logger = logging.getLogger("STDMS.Agent.RAG.Ingest")

DEFAULT_DOCS_DIR = Path("data") / "knowledge"
SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown", ".pdf"}
# Soft cap so a single huge PDF cannot stall rebuild for hours
MAX_CHUNKS_PER_DOCUMENT = 400
# Non-mission / personal docs live here — never indexed into ops RAG
_SKIP_DIR_NAMES = frozenset({"_archive", "_offtopic", ".git", "__pycache__"})
# Filename substrings for personal / non-mission PDFs still under knowledge/
_SKIP_NAME_SUBSTR = (
    "ielts",
    "magoosh",
    "ahmadzade",
    "bayramov",
    "profile_data",
    "vocabulary",
    "math fro com",
    "pid_lecture",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def read_document(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".markdown"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError:
            logger.warning("pypdf not installed — skip PDF %s (pip install pypdf)", path.name)
            return None
        try:
            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            return "\n".join(parts).strip() or None
        except Exception as exc:
            logger.warning("PDF extract failed for %s: %s", path, exc)
            return None
    return None


def ingest_knowledge_dir(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_fn: Optional[Callable[..., Optional[list[float]]]] = None,
    force: bool = False,
    source_prefix: str = "",
    skip_nested_segments: bool = False,
) -> dict[str, Any]:
    """Scan docs_dir, embed new/changed files, update SQLite index.

    ``source_prefix`` (e.g. ``space``) stores sources as ``space/file.md``.
    ``skip_nested_segments`` ignores ``space/`` and ``ground/`` children when
    scanning the knowledge root (used by legacy single-index rebuilds).
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(index_path)
    embed = embed_fn or embed_ollama

    if embed_fn is None and not ollama_reachable(ollama_url):
        return {
            "ok": False,
            "error": "ollama_unreachable",
            "note": f"Ollama not reachable at {ollama_url}. Start Ollama and pull {embed_model}.",
            "indexed": 0,
            "skipped": 0,
        }

    # Probe embedding model once (404 = model not pulled)
    if embed_fn is None:
        probe = embed("STDMS knowledge index probe", base_url=ollama_url, model=embed_model)
        if not probe:
            return {
                "ok": False,
                "error": "embed_model_unavailable",
                "note": (
                    f"Embedding model '{embed_model}' not available (Ollama 404). "
                    f"Run: ollama pull {embed_model}"
                ),
                "indexed": 0,
                "skipped": 0,
            }

    segment_dirs = frozenset({"space", "ground"})

    def _include_doc(p: Path) -> bool:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
            return False
        if any(part in _SKIP_DIR_NAMES for part in p.parts):
            return False
        if skip_nested_segments:
            try:
                rel_parts = p.relative_to(docs_path).parts
            except ValueError:
                rel_parts = p.parts
            if rel_parts and rel_parts[0].lower() in segment_dirs:
                return False
        name_l = p.name.lower()
        if any(s in name_l for s in _SKIP_NAME_SUBSTR):
            return False
        return True

    files = sorted(p for p in docs_path.rglob("*") if _include_doc(p))
    keep: set[str] = set()
    indexed = 0
    skipped = 0
    failed = 0
    errors: list[str] = []
    prefix = (source_prefix or "").strip().strip("/").replace("\\", "/")

    for path in files:
        try:
            rel_body = path.relative_to(docs_path).as_posix()
        except ValueError:
            rel_body = path.name
        rel = f"{prefix}/{rel_body}" if prefix else str(path.as_posix())
        keep.add(rel)
        text = read_document(path)
        if not text or not text.strip():
            skipped += 1
            continue
        content_hash = _sha256_text(text)
        if not force and store.get_document_hash(rel) == content_hash:
            skipped += 1
            continue

        pieces = chunk_text(text)
        if not pieces:
            skipped += 1
            continue
        if len(pieces) > MAX_CHUNKS_PER_DOCUMENT:
            logger.warning(
                "Truncating %s from %d to %d chunks (MAX_CHUNKS_PER_DOCUMENT)",
                path.name,
                len(pieces),
                MAX_CHUNKS_PER_DOCUMENT,
            )
            pieces = pieces[:MAX_CHUNKS_PER_DOCUMENT]

        chunk_rows: list[dict[str, Any]] = []
        ok = True
        for i, piece in enumerate(pieces):
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(pieces):
                logger.info("Embedding %s chunk %d/%d", path.name, i + 1, len(pieces))
            vec = embed(piece, base_url=ollama_url, model=embed_model)
            if not vec:
                ok = False
                errors.append(f"embed failed: {path.name}")
                break
            chunk_rows.append(
                {
                    "text": piece,
                    "embedding": vec,
                    "meta": {"chars": len(piece)},
                }
            )
        if not ok:
            failed += 1
            continue

        store.upsert_document_chunks(
            source_path=rel,
            title=path.stem.replace("_", " ").replace("-", " "),
            content_hash=content_hash,
            mtime=path.stat().st_mtime,
            chunks=chunk_rows,
            indexed_at=datetime.datetime.now().isoformat(),
        )
        indexed += 1
        logger.info("Indexed %s (%d chunks)", path.name, len(chunk_rows))

    removed = store.delete_missing_sources(keep)
    return {
        "ok": failed == 0,
        "docs_dir": str(docs_path),
        "index_path": str(index_path),
        "files_seen": len(files),
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "removed": removed,
        "documents": store.count_documents(),
        "chunks": store.count_chunks(),
        "errors": errors[:10],
        "embed_model": embed_model,
        "source_prefix": prefix or None,
    }


def export_fsm_to_knowledge(
    output_dir: str | Path = DEFAULT_DOCS_DIR / "space",
    *,
    tabs_config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    filename: str = "mission_modes_auto.md",
) -> dict[str, Any]:
    """Write per-tab mission modes into space knowledge Markdown."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = tabs_config
    if configs is None:
        cfg_path = Path(config_path or (Path("data") / "custom_tabs_config.json"))
        if cfg_path.is_file():
            import json

            try:
                configs = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("FSM export: failed to read %s: %s", cfg_path, exc)
                configs = {}
        else:
            configs = {}

    lines: list[str] = [
        "# Mission Modes (auto-exported from STDMS tab FSM)",
        "",
        "Generated for RAG space store. Do not edit by hand — Rebuild Knowledge refreshes this file.",
        "",
    ]
    exported_tabs = 0
    for tab_id, cfg in (configs or {}).items():
        if not isinstance(cfg, dict):
            continue
        title = str(cfg.get("title") or tab_id)
        current = str(cfg.get("current_mission_mode") or "nominal")
        modes = cfg.get("mission_modes") or []
        if not modes:
            continue
        exported_tabs += 1
        lines.append(f"# Tab: {title} — Mission Modes")
        lines.append(f"- tab_id: `{tab_id}`")
        lines.append(f"- current_mission_mode: **{current}**")
        lines.append("")
        for mode in modes:
            if not isinstance(mode, dict):
                continue
            name = str(mode.get("name") or "nominal")
            try:
                scale = float(mode.get("threshold_scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            desc = str(mode.get("description") or "").strip()
            lines.append(f"## {name} (threshold_scale: {scale:g})")
            if desc:
                lines.append(f"- {desc}")
            if name == "eclipse":
                lines.append("- TCS WARNING / thermal swing → mode-normal (expected in eclipse)")
                lines.append("- Solar panel / battery telemetry deviation may be expected")
            elif name == "nominal":
                lines.append("- All deviations against base thresholds are anomalies")
            elif name == "maneuver":
                lines.append("- ADCS / vibration transients → often mode-normal")
            elif name == "safe_mode":
                lines.append("- Tighter sensitivity (scale < 1); escalate warnings faster")
            lines.append("")
        lines.append("---")
        lines.append("")

    if exported_tabs == 0:
        lines.extend(
            [
                "# Default Mission Modes",
                "",
                "## nominal (threshold_scale: 1.0)",
                "- All deviations are anomalies",
                "",
                "## eclipse (threshold_scale: 1.5)",
                "- TCS WARNING → mode-normal",
                "- Solar panel telemetry deviation expected",
                "",
                "## maneuver (threshold_scale: 2.0)",
                "- ADCS vibration spikes often mode-normal",
                "",
                "## safe_mode (threshold_scale: 0.5)",
                "- Tighter anomaly sensitivity",
                "",
            ]
        )

    path = out_dir / filename
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "ok": True,
        "path": str(path),
        "tabs_exported": exported_tabs,
    }


def ensure_segment_dirs(docs_dir: str | Path = DEFAULT_DOCS_DIR) -> dict[str, Path]:
    root = Path(docs_dir)
    space = root / "space"
    ground = root / "ground"
    space.mkdir(parents=True, exist_ok=True)
    ground.mkdir(parents=True, exist_ok=True)
    return {"root": root, "space": space, "ground": ground}


def ingest_dual_knowledge(
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    embed_fn: Optional[Callable[..., Optional[list[float]]]] = None,
    force: bool = False,
    tabs_config: dict[str, Any] | None = None,
    export_fsm: bool = True,
) -> dict[str, Any]:
    """Index space/ and ground/ into separate SQLite stores + FSM auto-export."""
    from app.agent.rag.dual import GROUND_INDEX_PATH, SPACE_INDEX_PATH

    dirs = ensure_segment_dirs(docs_dir)
    fsm_info: dict[str, Any] = {}
    if export_fsm:
        fsm_info = export_fsm_to_knowledge(dirs["space"], tabs_config=tabs_config)

    space_result = ingest_knowledge_dir(
        dirs["space"],
        index_path=SPACE_INDEX_PATH,
        ollama_url=ollama_url,
        embed_model=embed_model,
        embed_fn=embed_fn,
        force=force,
        source_prefix="space",
    )
    ground_result = ingest_knowledge_dir(
        dirs["ground"],
        index_path=GROUND_INDEX_PATH,
        ollama_url=ollama_url,
        embed_model=embed_model,
        embed_fn=embed_fn,
        force=force,
        source_prefix="ground",
    )

    # Keep a legacy combined index of non-segment root docs for backward compatibility
    legacy = ingest_knowledge_dir(
        docs_dir,
        index_path=DEFAULT_INDEX_PATH,
        ollama_url=ollama_url,
        embed_model=embed_model,
        embed_fn=embed_fn,
        force=force,
        skip_nested_segments=True,
    )

    ok = bool(space_result.get("ok")) and bool(ground_result.get("ok"))
    # If Ollama down, both fail the same way — surface that error
    err = space_result.get("error") or ground_result.get("error") or legacy.get("error")
    return {
        "ok": ok if not err else False,
        "error": err,
        "note": space_result.get("note") or ground_result.get("note") or legacy.get("note"),
        "mode": "dual_space_ground",
        "fsm_export": fsm_info,
        "space": space_result,
        "ground": ground_result,
        "legacy": legacy,
        "indexed": int(space_result.get("indexed") or 0)
        + int(ground_result.get("indexed") or 0)
        + int(legacy.get("indexed") or 0),
        "documents": int(space_result.get("documents") or 0)
        + int(ground_result.get("documents") or 0),
        "chunks": int(space_result.get("chunks") or 0) + int(ground_result.get("chunks") or 0),
        "files_seen": int(space_result.get("files_seen") or 0)
        + int(ground_result.get("files_seen") or 0),
        "errors": (space_result.get("errors") or []) + (ground_result.get("errors") or []),
    }


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest data/knowledge into dual local RAG indexes")
    parser.add_argument("--docs", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH), help="Legacy single index (optional)")
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--ollama", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--legacy-only", action="store_true", help="Only build the single legacy index")
    args = parser.parse_args(argv)
    if args.legacy_only:
        result = ingest_knowledge_dir(
            args.docs,
            index_path=args.index,
            ollama_url=args.ollama,
            embed_model=args.model,
            force=args.force,
        )
    else:
        result = ingest_dual_knowledge(
            args.docs,
            ollama_url=args.ollama,
            embed_model=args.model,
            force=args.force,
        )
    print(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
