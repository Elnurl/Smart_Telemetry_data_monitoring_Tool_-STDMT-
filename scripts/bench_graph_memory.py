#!/usr/bin/env python3
"""Benchmark NetworkX graph memory recall (Sprint 2 portfolio numbers).

Usage:
    py -3.10 scripts/bench_graph_memory.py
    py -3.10 scripts/bench_graph_memory.py --tabs 50 --decisions 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.memory.networkx_store import NetworkXGraphStore
from app.agent.memory.sync import record_decision_memory


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench graph memory recall")
    parser.add_argument("--tabs", type=int, default=40)
    parser.add_argument("--decisions", type=int, default=15, help="Decisions per tab")
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "bench_graph.sqlite")
        store = NetworkXGraphStore(db_path=db)
        modes = ["nominal", "eclipse", "maneuver", "safe_mode"]
        t0 = time.perf_counter()
        for i in range(args.tabs):
            tid = f"tab-{i:04d}"
            mode = modes[i % len(modes)]
            store.upsert_node(
                "Tab",
                tid,
                {
                    "title": f"Subsystem-{i % 8}",
                    "health": "Warning" if i % 5 == 0 else "OK",
                    "mission_mode": mode,
                    "threshold_scale": 1.5 if mode == "eclipse" else 1.0,
                    "monitoring": True,
                },
            )
            store.upsert_edge("Tab", tid, "IN_MODE", "Mode", mode)
            if i % 3 == 0:
                store.upsert_node(
                    "Draft",
                    f"d-{i}",
                    {"kind": "alert", "severity": "WARNING", "status": "pending", "tab_id": tid},
                )
                store.upsert_edge("Tab", tid, "HAS_DRAFT", "Draft", f"d-{i}")
            for j in range(args.decisions):
                record_decision_memory(
                    store,
                    decision_id=f"{i}-{j}",
                    tab_id=tid,
                    outcome="ok",
                    tool_called="get_fleet_status",
                    reasoning=f"cycle {j}",
                    tool_trace=["get_fleet_status"],
                )
        build_ms = (time.perf_counter() - t0) * 1000.0

        queries = [
            "warning eclipse thermal",
            "pending draft alert",
            "retrain drift",
            "subsystem fleet status",
            "maneuver mode",
        ]
        latencies: list[float] = []
        for n in range(args.queries):
            q = queries[n % len(queries)]
            tid = f"tab-{(n % args.tabs):04d}"
            start = time.perf_counter()
            store.recall(q, tab_id=tid, limit=12)
            latencies.append((time.perf_counter() - start) * 1000.0)

        latencies.sort()
        st = store.stats()
        print("=== STDMS GraphMemory (NetworkX) bench ===")
        print(f"tabs={args.tabs} decisions_per_tab={args.decisions} queries={args.queries}")
        print(f"build_ms={build_ms:.2f}")
        print(f"nodes={st['nodes']} edges={st['edges']} types={st['node_types']}")
        print(
            "recall_ms: p50={:.3f}  p95={:.3f}  mean={:.3f}".format(
                _percentile(latencies, 50),
                _percentile(latencies, 95),
                statistics.fmean(latencies) if latencies else 0.0,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
