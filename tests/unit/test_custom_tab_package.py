"""Smoke tests for extracted custom monitoring tab workers (A3)."""

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QThread

from app.tabs.custom_tab.workers import TabMonitoringWorker, TabTrainWorker


def test_worker_classes_are_qthreads():
    assert issubclass(TabMonitoringWorker, QThread)
    assert issubclass(TabTrainWorker, QThread)


def test_worker_signal_attributes():
    assert hasattr(TabMonitoringWorker, "finished")
    assert hasattr(TabTrainWorker, "finished")
