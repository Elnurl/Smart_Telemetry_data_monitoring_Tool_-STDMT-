"""Unit tests for AsyncMonitoringPipeline (Sprint 1)."""

from __future__ import annotations

import threading
import time

import pytest

from app.monitoring.async_pipeline import (
    AsyncMonitoringPipeline,
    reset_async_pipeline_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_async_pipeline_for_tests()
    yield
    reset_async_pipeline_for_tests()


def test_two_tabs_run_overlapping_cycles():
    pipe = AsyncMonitoringPipeline(max_workers=4)
    pipe.start()
    barrier = threading.Barrier(2)
    done = {"a": 0, "b": 0}
    lock = threading.Lock()

    def make_compute(tab: str):
        def _fn():
            barrier.wait(timeout=2.0)
            with lock:
                done[tab] += 1
            return {"status": "ok", "tab": tab}

        return _fn

    results = []
    ready = threading.Event()

    def on_result(tab_id, result, metrics):
        results.append((tab_id, result, metrics.status))
        if len(results) >= 2:
            ready.set()

    pipe.register_tab(
        "tab-a",
        interval_ms=5000,
        compute_fn=make_compute("a"),
        on_result=on_result,
        run_immediately=True,
    )
    pipe.register_tab(
        "tab-b",
        interval_ms=5000,
        compute_fn=make_compute("b"),
        on_result=on_result,
        run_immediately=True,
    )
    assert ready.wait(timeout=5.0), "tabs did not complete overlapping cycles"
    assert done["a"] == 1 and done["b"] == 1
    statuses = {r[2] for r in results}
    assert "ok" in statuses
    pipe.stop()


def test_overlap_skip_when_cycle_still_running():
    pipe = AsyncMonitoringPipeline(max_workers=2)
    pipe.start()
    entered = threading.Event()
    release = threading.Event()
    metrics_seen = []
    done = threading.Event()

    def compute():
        entered.set()
        release.wait(timeout=5.0)
        return {"status": "ok"}

    def on_result(tab_id, result, metrics):
        metrics_seen.append(metrics)
        if metrics.status == "ok":
            done.set()

    pipe.register_tab(
        "slow",
        interval_ms=50,
        compute_fn=compute,
        on_result=on_result,
        run_immediately=True,
    )
    assert entered.wait(timeout=2.0)
    # Allow several ticks while still busy → skipped_overlap metrics
    time.sleep(0.25)
    release.set()
    assert done.wait(timeout=3.0)
    # Wait for skip metrics to be recorded
    deadline = time.time() + 2.0
    while time.time() < deadline:
        recent = pipe.get_recent_metrics(50)
        if any(m["status"] == "skipped_overlap" for m in recent):
            break
        time.sleep(0.05)
    recent = pipe.get_recent_metrics(50)
    assert any(m["status"] == "skipped_overlap" for m in recent), recent
    pipe.stop()


def test_metrics_fields_populated():
    pipe = AsyncMonitoringPipeline(max_workers=2)
    pipe.start()
    ready = threading.Event()

    def compute():
        time.sleep(0.05)
        return {"status": "ok"}

    def on_result(tab_id, result, metrics):
        ready.set()

    pipe.submit_once("m1", compute, on_result=on_result)
    assert ready.wait(timeout=3.0)
    time.sleep(0.05)
    recent = pipe.get_recent_metrics(10)
    assert recent, "expected metrics"
    m = recent[-1]
    assert m["tab_id"] == "m1"
    assert m["status"] == "ok"
    assert m["compute_ms"] >= 40
    assert m["total_ms"] >= m["compute_ms"] - 1
    assert "queue_wait_ms" in m
    pipe.stop()


def test_unregister_stops_further_ticks():
    pipe = AsyncMonitoringPipeline(max_workers=2)
    pipe.start()
    count = {"n": 0}
    lock = threading.Lock()

    def compute():
        with lock:
            count["n"] += 1
        return {"status": "ok"}

    pipe.register_tab(
        "stop-me",
        interval_ms=80,
        compute_fn=compute,
        run_immediately=True,
    )
    time.sleep(0.2)
    pipe.unregister_tab("stop-me")
    time.sleep(0.1)
    with lock:
        after_unreg = count["n"]
    time.sleep(0.25)
    with lock:
        final = count["n"]
    assert final == after_unreg, f"ticks continued after unregister: {after_unreg} -> {final}"
    assert "stop-me" not in pipe.active_tab_ids()
    pipe.stop()
