"""Tests for configure_custom_tab injection and guards (A1/A2 + B6 panel slots).

A2 coverage: ``test_unconfigured_custom_tab_raises``, ``test_incomplete_configure_raises``.
B6 Phase 3: ``build_legacy_panel_slots`` factory + ``assert_panel_slots_complete``.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QWidget


@pytest.fixture(scope="session")
def qapplication():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def widget_module():
    import app.tabs.custom_tab.widget as widget_module

    snapshot = {key: getattr(widget_module, key) for key in widget_module._CONFIG_KEYS}
    snapshot["_CONFIGURED"] = widget_module._CONFIGURED
    yield widget_module
    for key, value in snapshot.items():
        setattr(widget_module, key, value)


def _fake_deps(widget_module, enhanced_cls):
    class _Tool:
        pass

    class _DataProcessor:
        def __init__(self):
            self.data = None
            self.preprocessed_data = None
            self.timestamp_column = None

    class _Model:
        def __init__(self, model_type):
            self.model_type = model_type

        def train(self, data, **kwargs):
            return True, "ok"

    class _Dialog:
        pass

    def _fake_slots_factory(_tool):
        from app.tabs.custom_tab.panels.slots import LegacyPanelSlots

        def _slot(_host, _method):
            return lambda *args, **kwargs: None

        return LegacyPanelSlots(
            tool_window=_tool,
            ensure_panel_helpers=lambda _h: None,
            data_slot=_slot,
            data_import_stop_slot=lambda _h: (lambda: None),
            analysis_slot=_slot,
            viz_slot=_slot,
            invoke_analysis_method=lambda *args, **kwargs: None,
            invoke_visualization_method=lambda *args, **kwargs: None,
            get_supported_model_names=lambda: ["Isolation Forest"],
            mpl_canvas_cls=object,
        )

    widget_module.configure_custom_tab(
        DATA_DIR="/tmp/data",
        REPORTS_DIR="/tmp/reports",
        SecureAnomalyDetectionTool=_Tool,
        DataProcessor=_DataProcessor,
        AnomalyDetectionModel=_Model,
        EnhancedAnomalyDetectionModel=enhanced_cls,
        MplCanvas=object,
        format_registry_error=lambda exc: str(exc),
        to_internal_model_type=lambda name: name.lower().replace(" ", "_"),
        AddModelDialog=_Dialog,
        TabConfigurationDialog=_Dialog,
        safe_pickle_load=lambda path: None,
        build_legacy_panel_slots=_fake_slots_factory,
    )


def test_configure_sets_enhanced_model(widget_module):
    class FakeEnhanced:
        def __init__(self, model_type):
            self.model_type = model_type

        @classmethod
        def load(cls, filepath):
            return cls("loaded"), "ok"

    _fake_deps(widget_module, FakeEnhanced)
    instance = widget_module.EnhancedAnomalyDetectionModel("enhanced_isolation_forest")
    assert instance.model_type == "enhanced_isolation_forest"
    loaded, message = widget_module.EnhancedAnomalyDetectionModel.load("model.pkl")
    assert loaded.model_type == "loaded"
    assert message == "ok"


def test_unconfigured_custom_tab_raises(widget_module, qapplication):
    widget_module._CONFIGURED = False
    for key in widget_module._CONFIG_KEYS:
        setattr(widget_module, key, None)

    with pytest.raises(RuntimeError, match="call configure_custom_tab\\(\\) first"):
        widget_module.CustomMonitoringTab(
            "tab-1",
            {"title": "Test"},
            QWidget(),
        )


def test_incomplete_configure_raises(widget_module):
    with pytest.raises(RuntimeError, match="configure_custom_tab\\(\\) incomplete"):
        widget_module.configure_custom_tab(DATA_DIR="/tmp/data")


def test_assert_panel_slots_complete_rejects_none():
    from app.tabs.custom_tab.panels.slots import assert_panel_slots_complete

    with pytest.raises(RuntimeError, match="not configured"):
        assert_panel_slots_complete(None)


def test_assert_panel_slots_complete_accepts_bundle():
    from app.tabs.custom_tab.panels.slots import LegacyPanelSlots, assert_panel_slots_complete

    slots = LegacyPanelSlots(
        tool_window=object(),
        ensure_panel_helpers=lambda h: None,
        data_slot=lambda h, m: lambda: None,
        data_import_stop_slot=lambda h: lambda: None,
        analysis_slot=lambda h, m: lambda: None,
        viz_slot=lambda h, m: lambda: None,
        invoke_analysis_method=lambda *a, **k: None,
        invoke_visualization_method=lambda *a, **k: None,
        get_supported_model_names=lambda: [],
        mpl_canvas_cls=object,
    )
    assert assert_panel_slots_complete(slots) is slots
