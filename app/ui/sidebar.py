from __future__ import annotations

from app.ui.qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ActiveTabsSidebar(QWidget):
    tab_selected = None  # set by main window to pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Active Tabs")
        title.setStyleSheet("font-weight: 700; font-size: 14px;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("sidebarList")
        layout.addWidget(self.list_widget, 1)

        hint = QLabel("Select a tab to open monitoring view.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5c6478; font-size: 12px;")
        layout.addWidget(hint)

    def set_tabs(self, tabs: list[tuple[str, str, str]]) -> None:
        """tabs: list of (tab_id, title, status_text)"""
        current_item = self.list_widget.currentItem()
        current = current_item.data(256) if current_item else None
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for tab_id, title, status in tabs:
            item = QListWidgetItem(f"{title}\n{status}")
            item.setData(256, tab_id)  # Qt.UserRole = 256
            self.list_widget.addItem(item)
            if tab_id == current:
                self.list_widget.setCurrentItem(item)
        self.list_widget.blockSignals(False)

    def current_tab_id(self) -> str | None:
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(256)
