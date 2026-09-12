#!/usr/bin/env python3
"""Benchmark AsyncMonitoringPipeline latency (Sprint 1 portfolio numbers).

Usage:
    py -3.10 scripts/bench_async_pipeline.py
    py -3.10 scripts/bench_async_pipeline.py --tabs 4 --cycles 20 --sleep-ms 25
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.monitoring.async_pipeline import AsyncMonitoringPipeline, reset_async_pipeline_for_tests


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
    parser = argparse.ArgumentParser(description="Bench AsyncMonitoringPipeline")
    parser.add_argument("--tabs", type=int, default=4, help="Number of fake tabs")
    parser.add_argument("--cycles", type=int, default=15, help="Successful cycles per tab")
    parser.add_argument("--sleep-ms", type=float, default=30.0, help="Fake compute sleep per cycle")
    parser.add_argument("--interval-ms", type=int, default=50, help="Schedule interval")
    parser.add_argument("--workers", type=int, default=0, help="Executor workers (0=default)")
    args = parser.parse_args()

    reset_async_pipeline_for_tests()
    max_workers = args.workers or None
    pipe = AsyncMonitoringPipeline(max_workers=max_workers)
    pipe.start()

    target_ok = args.tabs * args.cycles
    ok_lock = threading.Lock()
    ok_count = {"n": 0}
    wall_start = time.perf_counter()
    done = threading.Event()

    def make_compute(sleep_s: float):
        def _fn():
            time.sleep(sleep_s)
            return {"status": "ok"}

        return _fn

    def on_result(tab_id, result, metrics):
        if metrics.status != "ok":
            return
        with ok_lock:
            ok_count["n"] += 1
            if ok_count["n"] >= target_ok:
                done.set()

    sleep_s = max(0.0, args.sleep_ms / 1000.0)
    for i in range(args.tabs):
        tid = f"bench-{i}"
        pipe.register_tab(
            tid,
            interval_ms=args.interval_ms,
            compute_fn=make_compute(sleep_s),
            on_result=on_result,
            schedule_type="Continuous",
            run_immediately=True,
        )

    timeout = max(30.0, (sleep_s + 0.05) * args.cycles * args.tabs + 10.0)
    if not done.wait(timeout=timeout):
        print("ERROR: benchmark timed out", file=sys.stderr)
        pipe.stop()
        return 1

    wall_s = time.perf_counter() - wall_start
    # Unregister so no more ticks pollute metrics
    for i in range(args.tabs):
        pipe.unregister_tab(f"bench-{i}")
    time.sleep(0.05)

    metrics = [m for m in pipe.get_recent_metrics(2000) if m["status"] == "ok"]
    compute = sorted(float(m["compute_ms"]) for m in metrics)
    total = sorted(float(m["total_ms"]) for m in metrics)
    queue = sorted(float(m["queue_wait_ms"]) for m in metrics)

    print("=== STDMS AsyncMonitoringPipeline bench ===")
    print(f"tabs={args.tabs} target_ok_cycles={target_ok} sleep_ms={args.sleep_ms}")
    print(f"workers={pipe._max_workers} wall_s={wall_s:.3f}")
    print(f"ok_cycles_recorded={len(metrics)} throughput_cycles_per_s={len(metrics) / wall_s:.2f}")
    print()
    print("compute_ms:  p50={:.2f}  p95={:.2f}  mean={:.2f}".format(
        _percentile(compute, 50),
        _percentile(compute, 95),
        statistics.fmean(compute) if compute else 0.0,
    ))
    print("total_ms:    p50={:.2f}  p95={:.2f}  mean={:.2f}".format(
        _percentile(total, 50),
        _percentile(total, 95),
        statistics.fmean(total) if total else 0.0,
    ))
    print("queue_wait_ms: p50={:.2f}  p95={:.2f}".format(
        _percentile(queue, 50),
        _percentile(queue, 95),
    ))
    skipped = sum(1 for m in pipe.get_recent_metrics(2000) if m["status"] == "skipped_overlap")
    print(f"skipped_overlap={skipped}")

    pipe.stop()
    reset_async_pipeline_for_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
