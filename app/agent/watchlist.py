"""Persistent agent watchlist (Faza F)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DEFAULT_WATCHLIST_PATH = Path("data") / "agent_watchlist.json"


def _path(root: Optional[Path | str] = None) -> Path:
    return Path(root) if root is not None else DEFAULT_WATCHLIST_PATH


def load_watchlist(*, path: Optional[Path | str] = None) -> dict[str, Any]:
    p = _path(path)
    if not p.is_file():
        return {"ok": True, "tabs": [], "path": str(p)}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        tabs = data.get("tabs") if isinstance(data, dict) else []
        if not isinstance(tabs, list):
            tabs = []
        cleaned = []
        for t in tabs:
            if isinstance(t, dict) and t.get("tab_id"):
                cleaned.append(
                    {
                        "tab_id": str(t["tab_id"]),
                        "title": str(t.get("title") or ""),
                        "note": str(t.get("note") or ""),
                    }
                )
            elif isinstance(t, str):
                cleaned.append({"tab_id": t, "title": "", "note": ""})
        return {"ok": True, "tabs": cleaned, "path": str(p)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "tabs": [], "path": str(p)}


def save_watchlist(tabs: list[dict[str, Any]], *, path: Optional[Path | str] = None) -> dict[str, Any]:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tabs": tabs}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "tabs": tabs, "path": str(p), "count": len(tabs)}


def add_to_watchlist(
    tab_id: str,
    *,
    title: str = "",
    note: str = "",
    path: Optional[Path | str] = None,
) -> dict[str, Any]:
    cur = load_watchlist(path=path)
    tabs = list(cur.get("tabs") or [])
    tid = str(tab_id).strip()
    if not tid:
        return {"ok": False, "error": "tab_id_required"}
    tabs = [t for t in tabs if t.get("tab_id") != tid]
    tabs.insert(0, {"tab_id": tid, "title": title or "", "note": note or ""})
    return save_watchlist(tabs, path=path)


def remove_from_watchlist(tab_id: str, *, path: Optional[Path | str] = None) -> dict[str, Any]:
    cur = load_watchlist(path=path)
    tid = str(tab_id).strip()
    tabs = [t for t in (cur.get("tabs") or []) if t.get("tab_id") != tid]
    return save_watchlist(tabs, path=path)


def pending_actions_summary(host: Any) -> dict[str, Any]:
    """Short summary of pending drafts/retrain for chat UX."""
    try:
        drafts = host.list_pending_drafts() if host else []
    except Exception:
        drafts = []
    try:
        retrains = host.get_pending_retrain_signals() if host else []
    except Exception:
        retrains = []
    watch = load_watchlist()
    lines = []
    if drafts:
        lines.append(f"{len(drafts)} pending agent draft(s) awaiting Approve/Reject")
        for d in drafts[:5]:
            lines.append(
                f"  - #{d.get('id')} {d.get('kind')} tab={d.get('tab_id')} sev={d.get('severity')}: "
                f"{str(d.get('proposed_message') or '')[:80]}"
            )
    else:
        lines.append("No pending agent drafts")
    if retrains:
        lines.append(f"{len(retrains)} pending retrain signal(s)")
    lines.append(f"Watchlist tabs: {len(watch.get('tabs') or [])}")
    return {
        "ok": True,
        "pending_draft_count": len(drafts or []),
        "pending_retrain_count": len(retrains or []),
        "watchlist_count": len(watch.get("tabs") or []),
        "summary_lines": lines,
        "summary": "\n".join(lines),
    }
