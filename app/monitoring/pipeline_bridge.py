"""Qt bridge for AsyncMonitoringPipeline → GUI thread result delivery."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from app.monitoring.async_pipeline import (
    AsyncMonitoringPipeline,
    CycleMetrics,
    get_async_pipeline,
)

logger = logging.getLogger("STDMS.AsyncPipeline")


class PipelineQtBridge(QObject):
    """Marshals cycle results onto the Qt GUI thread via pyqtSignal."""

    cycle_finished = pyqtSignal(str, object, object)  # tab_id, result, metrics

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._handlers: dict[str, Callable[[Any, CycleMetrics], None]] = {}
        self._lock = threading.Lock()
        self.cycle_finished.connect(self._on_cycle_finished)

    def set_handler(
        self,
        tab_id: str,
        handler: Callable[[Any, CycleMetrics], None],
    ) -> None:
        with self._lock:
            self._handlers[tab_id] = handler

    def clear_handler(self, tab_id: str) -> None:
        with self._lock:
            self._handlers.pop(tab_id, None)

    def emit_result(self, tab_id: str, result: Any, metrics: CycleMetrics) -> None:
        """Thread-safe delivery onto the GUI thread.

        Emit the QueuedConnection signal directly. Do **not** use
        ``QTimer.singleShot`` from worker/async threads — those timers are
        owned by the calling thread and often never fire, so Health Status
        stays N/A even when cycles succeed.
        """
        self.cycle_finished.emit(tab_id, result, metrics)

    def _on_cycle_finished(self, tab_id: str, result: object, metrics: object) -> None:
        with self._lock:
            handler = self._handlers.get(tab_id)
        if handler is None:
            return
        try:
            handler(result, metrics)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("PipelineQtBridge handler failed for %s: %s", tab_id, exc)


_BRIDGE: Optional[PipelineQtBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_pipeline_bridge() -> PipelineQtBridge:
    """GUI-thread singleton (create from main window / first custom tab)."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = PipelineQtBridge()
        return _BRIDGE


def reset_pipeline_bridge_for_tests() -> None:
    global _BRIDGE
    with _BRIDGE_LOCK:
        _BRIDGE = None


def make_on_result_callback(bridge: Optional[PipelineQtBridge] = None):
    """Return pipeline on_result that posts to the Qt bridge."""

    def _on_result(tab_id: str, result: Any, metrics: CycleMetrics) -> None:
        b = bridge or get_pipeline_bridge()
        b.emit_result(tab_id, result, metrics)

    return _on_result


def ensure_pipeline() -> AsyncMonitoringPipeline:
    pipe = get_async_pipeline()
    if not pipe.is_running:
        pipe.start()
    return pipe
