"""Navigation shell for the custom monitoring tab (Yamcs-style sidebar + icons)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.ui.nav_icons import CUSTOM_TAB_NAV_ENTRIES, nav_icon, nav_icon_size

if TYPE_CHECKING:
    from app.tabs.custom_tab.widget import CustomMonitoringTab

_SIDEBAR_STYLE = """
#yamcsSidebar {
    background: #f0f2f5;
    border-right: 1px solid #dfe3eb;
}
#yamcsNavSearch {
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 4px;
    padding: 6px 8px;
    margin: 4px 2px 8px 2px;
}
#yamcsNavList {
    background: transparent;
    border: none;
    outline: none;
}
#yamcsNavList::item {
    padding: 10px 12px;
    border-radius: 4px;
    margin: 2px 4px;
    color: #1a1d26;
}
#yamcsNavList::item:selected {
    background: #d8e4f0;
    color: #1a1d26;
}
#yamcsNavList::item:hover {
    background: #e8eef5;
}
"""


def _filter_nav_list(nav_list: QListWidget, text: str) -> None:
    needle = text.strip().lower()
    for row in range(nav_list.count()):
        item = nav_list.item(row)
        if item is None:
            continue
        item.setHidden(bool(needle) and needle not in item.text().lower())


def build_nav_shell(tab: CustomMonitoringTab, root_layout: QHBoxLayout) -> None:
    """Create icon sidebar and stacked pages on *tab*."""
    nav_widget = QWidget()
    nav_widget.setObjectName("yamcsSidebar")
    nav_widget.setFixedWidth(220)
    nav_layout = QVBoxLayout(nav_widget)
    nav_layout.setContentsMargins(8, 8, 8, 8)
    nav_layout.setSpacing(4)

    search = QLineEdit()
    search.setObjectName("yamcsNavSearch")
    search.setPlaceholderText("Search sections")
    search.setClearButtonEnabled(True)
    nav_layout.addWidget(search)

    tab.custom_nav_list = QListWidget()
    tab.custom_nav_list.setObjectName("yamcsNavList")
    tab.custom_nav_list.setIconSize(nav_icon_size())
    tab.custom_nav_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    for label, icon_key in CUSTOM_TAB_NAV_ENTRIES:
        item = QListWidgetItem(nav_icon(tab, icon_key), label)
        item.setSizeHint(item.sizeHint())
        tab.custom_nav_list.addItem(item)

    search.textChanged.connect(lambda text: _filter_nav_list(tab.custom_nav_list, text))
    nav_layout.addWidget(tab.custom_nav_list, 1)
    nav_widget.setStyleSheet(_SIDEBAR_STYLE)
    root_layout.addWidget(nav_widget)

    tab.custom_pages = QStackedWidget()
    tab.custom_pages.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    root_layout.addWidget(tab.custom_pages, 1)


def connect_nav_pages(tab: CustomMonitoringTab) -> None:
    """Wire nav list to stacked widget after all pages are registered."""
    tab.custom_nav_list.currentRowChanged.connect(tab.custom_pages.setCurrentIndex)
    tab.custom_nav_list.setCurrentRow(0)
