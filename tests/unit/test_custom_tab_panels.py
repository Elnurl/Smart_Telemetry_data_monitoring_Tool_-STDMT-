"""Smoke tests for custom tab panel builders (B6 Phase 0–2)."""

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QGroupBox, QTextEdit, QWidget

from app.tabs.custom_tab.panels import (
    build_header_config_group,
    build_models_page,
    build_nav_shell,
    build_quick_actions_page,
    connect_nav_pages,
)
from app.tabs.custom_tab.panels.quick_actions import build_monitoring_log_group
from app.tabs.custom_tab.model_ops import model_types_equivalent
from app.tabs.custom_tab.panels.analysis_ml import _wire_analysis_train_button
from app.tabs.custom_tab.panels.slots import LegacyPanelSlots


@pytest.fixture(scope="session")
def qapplication():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_panel_builder_exports():
    for fn in (
        build_header_config_group,
        build_models_page,
        build_quick_actions_page,
        build_nav_shell,
        connect_nav_pages,
    ):
        assert callable(fn)


def test_build_monitoring_log_group_attaches_results_text(qapplication):
    class _Host(QWidget):
        pass

    host = _Host()
    group = build_monitoring_log_group(host)

    assert isinstance(group, QGroupBox)
    assert hasattr(host, "results_text")
    assert isinstance(host.results_text, QTextEdit)
    assert host.results_text.isReadOnly()
    assert host.results_text.minimumHeight() == 100
    assert host.results_text.placeholderText() == "Monitoring events will appear here."


def test_model_types_equivalent_matches_display_and_internal_keys():
    def _to_internal(name: str) -> str:
        table = {"XGBoost RUL": "xgboost_rul", "LSTM": "lstm"}
        return table.get(name, str(name).strip().lower().replace(" ", "_"))

    assert model_types_equivalent("xgboost_rul", "XGBoost RUL", _to_internal)
    assert model_types_equivalent("LSTM", "lstm", _to_internal)
    assert not model_types_equivalent("ARIMA", "LSTM", _to_internal)


def test_analysis_train_button_uses_canonical_redirect(qapplication):
    class _Host:
        def train_from_analysis_panel(self):
            self.called = True

    host = _Host()
    slots = LegacyPanelSlots(
        tool_window=object(),
        ensure_panel_helpers=lambda h: None,
        data_slot=lambda h, m: lambda: None,
        data_import_stop_slot=lambda h: lambda: None,
        analysis_slot=lambda h, m: (lambda: (_ for _ in ()).throw(AssertionError("legacy train"))),
        viz_slot=lambda h, m: lambda: None,
        invoke_analysis_method=lambda *a, **k: None,
        invoke_visualization_method=lambda *a, **k: None,
        get_supported_model_names=lambda: [],
        mpl_canvas_cls=object,
    )
    from PyQt5.QtWidgets import QPushButton

    btn = QPushButton("Train")
    _wire_analysis_train_button(btn, host, slots)
    btn.click()
    assert host.called is True
