"""PipelineQtBridge must deliver cycle results from worker threads to GUI."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication

from app.monitoring.async_pipeline import CycleMetrics
from app.monitoring.pipeline_bridge import (
    PipelineQtBridge,
    make_on_result_callback,
    reset_pipeline_bridge_for_tests,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _reset_bridge():
    reset_pipeline_bridge_for_tests()
    yield
    reset_pipeline_bridge_for_tests()


def test_emit_result_from_worker_thread_reaches_handler(qapp):
    bridge = PipelineQtBridge()
    received = []
    done = threading.Event()

    def handler(result, metrics):
        received.append((result, metrics.status if metrics else None))
        done.set()

    bridge.set_handler("tab-1", handler)
    metrics = CycleMetrics(tab_id="tab-1", started_at=time.time(), status="ok")

    def worker():
        cb = make_on_result_callback(bridge)
        cb("tab-1", {"status": "success", "fused_score": 0.12}, metrics)

    threading.Thread(target=worker, daemon=True).start()

    deadline = time.time() + 3.0
    while time.time() < deadline and not done.is_set():
        qapp.processEvents()
        time.sleep(0.01)

    assert done.is_set(), "GUI handler never received worker cycle result"
    assert received[0][0]["fused_score"] == 0.12
    assert received[0][1] == "ok"
