"""Model Development page (tab-native train/save/load — see panels package docstring)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from app.tabs.custom_tab.widget import CustomMonitoringTab


def build_models_page(tab: CustomMonitoringTab) -> QWidget:
    """Build Model Development page; sets train/save/load widgets on *tab*."""
    page_models = QWidget()
    page_models_layout = QVBoxLayout(page_models)
    page_models_layout.setSpacing(12)
    page_models_layout.setContentsMargins(10, 10, 10, 10)

    model_toolbar = QHBoxLayout()
    add_model_btn = QPushButton("+ Add Model")
    add_model_btn.clicked.connect(tab.add_model)
    model_toolbar.addWidget(add_model_btn)
    model_toolbar.addStretch()
    page_models_layout.addLayout(model_toolbar)

    tab.models_table = QTableWidget()
    tab.models_table.setColumnCount(5)
    tab.models_table.setHorizontalHeaderLabels(["Model", "Status", "Parameters", "Select", "Remove"])
    tab.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tab.models_table.setAlternatingRowColors(True)
    tab.models_table.verticalHeader().setVisible(False)
    tab.models_table.setMinimumHeight(240)
    page_models_layout.addWidget(tab.models_table, 1)
    tab.update_models_table()

    model_buttons_layout = QHBoxLayout()
    tab.train_btn = QPushButton("Train Selected")
    tab.train_btn.clicked.connect(tab.train_selected_model)
    tab.train_all_btn = QPushButton("Train All")
    tab.train_all_btn.clicked.connect(tab.train_all_models)
    tab.edit_model_params_btn = QPushButton("Edit Params")
    tab.edit_model_params_btn.clicked.connect(tab.edit_selected_model_params)
    tab.save_model_btn = QPushButton("Save")
    tab.save_model_btn.clicked.connect(tab.save_selected_model)
    tab.save_model_btn.setEnabled(False)
    tab.load_model_btn = QPushButton("Load")
    tab.load_model_btn.clicked.connect(tab.load_model)
    tab.recent_models_btn = QPushButton("Trained Models")
    tab.recent_models_btn.clicked.connect(tab.show_recent_models)
    tab.model_metrics_btn = QPushButton("Metrics")
    tab.model_metrics_btn.clicked.connect(tab.show_model_metrics)
    tab.model_metrics_btn.setEnabled(False)

    for btn in (
        tab.train_btn,
        tab.train_all_btn,
        tab.edit_model_params_btn,
        tab.save_model_btn,
        tab.load_model_btn,
        tab.recent_models_btn,
        tab.model_metrics_btn,
    ):
        model_buttons_layout.addWidget(btn)
    model_buttons_layout.addStretch()
    page_models_layout.addLayout(model_buttons_layout)

    tab.models_selection_label = QLabel("No model selected.")
    tab.models_selection_label.setWordWrap(True)
    page_models_layout.addWidget(tab.models_selection_label)

    return page_models
