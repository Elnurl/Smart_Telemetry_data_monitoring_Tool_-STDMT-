"""Background workers for custom monitoring tabs."""

from __future__ import annotations

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from app.ui.training_console import TrainingConsoleBridge, capture_stdout

logger = logging.getLogger("STDMS.CustomMonitoringTab")


class TabMonitoringWorker(QThread):
    """Run a custom-tab monitoring cycle off the Qt UI thread."""
    finished = pyqtSignal(object)

    def __init__(self, tab):
        super().__init__(tab)
        self.tab = tab

    def run(self):
        try:
            result = self.tab._compute_monitoring_cycle()
        except Exception as exc:
            logger.error("Monitoring worker failed: %s", exc)
            result = {"status": "error", "error": str(exc), "log_lines": []}
        self.finished.emit(result)


class TabTrainWorker(QThread):
    """Train a custom-tab model off the Qt UI thread."""
    finished = pyqtSignal(bool, str, str, object)
    log_chunk = pyqtSignal(str)

    def __init__(self, tab, model_id, data, model_obj, model_type, model_params):
        super().__init__(tab)
        self.tab = tab
        self.model_id = model_id
        self.data = data
        self.model_obj = model_obj
        self.model_type = model_type
        self.model_params = model_params or {}
        self._console_bridge = TrainingConsoleBridge()
        self._console_bridge.chunk.connect(self.log_chunk)

    def run(self):
        try:
            with capture_stdout(self._console_bridge):
                success, message = self.model_obj.train(self.data, **self.model_params)
            self.finished.emit(success, message, self.model_id, self.model_obj if success else None)
        except Exception as exc:
            logger.error("Tab training worker failed: %s", exc)
            self.finished.emit(False, str(exc), self.model_id, None)
