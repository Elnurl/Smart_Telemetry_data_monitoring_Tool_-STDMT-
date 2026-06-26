"""Offline HTML mission report generation."""

from __future__ import annotations

import datetime
import html
from pathlib import Path
from typing import Dict, Mapping


def export_mission_report(
    reports_dir: Path | str,
    title: str,
    snapshot: Mapping[str, object],
    anomaly_events: list[dict],
    *,
    max_events: int = 100,
) -> str:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() else "_" for c in title) or "tab"
    report_file = reports_dir / f"mission_{safe_title}_{timestamp}.html"

    snap_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in snapshot.items()
    )
    event_rows = ""
    for ev in anomaly_events[:max_events]:
        event_rows += (
            f"<tr><td>{html.escape(str(ev.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(ev.get('health_state', '')))}</td>"
            f"<td>{float(ev.get('fused_score', 0) or 0):.4f}</td>"
            f"<td>{html.escape(str(ev.get('xai', '')))}</td></tr>"
        )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mission Report — {html.escape(title)}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 32px; color: #1a1d26; }}
h1 {{ color: #4f6ef7; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #dfe3eb; padding: 8px 12px; text-align: left; }}
th {{ background: #f0f3f9; }}
.meta {{ color: #5c6478; }}
</style></head><body>
<h1>Mission Report: {html.escape(title)}</h1>
<p class="meta">Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Local offline operation</p>
<h2>Snapshot</h2>
<table>{snap_rows}</table>
<h2>Recent Anomaly Events</h2>
<table><tr><th>Time</th><th>State</th><th>Fusion Score</th><th>Explanation</th></tr>{event_rows or '<tr><td colspan=4>No events recorded.</td></tr>'}</table>
</body></html>"""

    report_file.write_text(doc, encoding="utf-8")
    return str(report_file)


def export_fleet_mission_report(
    reports_dir: Path | str,
    tab_snapshots: Dict[str, Mapping[str, object]],
) -> str:
    """Export combined HTML mission report for multiple monitoring tabs."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"fleet_mission_report_{timestamp}.html"

    sections = []
    for tab_id, snap in tab_snapshots.items():
        title = html.escape(str(snap.get("title", tab_id)))
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in snap.items()
        )
        sections.append(f"<h2>{title}</h2><table>{rows}</table>")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Fleet Mission Report</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 32px; color: #1a1d26; }}
h1 {{ color: #4f6ef7; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #dfe3eb; padding: 8px 12px; text-align: left; }}
.meta {{ color: #5c6478; }}
</style></head><body>
<h1>Fleet Mission Report</h1>
<p class="meta">Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Local offline operation</p>
{''.join(sections) or '<p>No tab snapshots available.</p>'}
</body></html>"""

    report_file.write_text(doc, encoding="utf-8")
    return str(report_file)
