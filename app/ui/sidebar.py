"""Spaceit-style dark ops navigation. STDMS functions stay on the right."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "azercosmos-logo.png"

NAV_QSS = """
QWidget#opsNav, QFrame#opsNavInner {
    background-color: #2c313a;
    color: #f3f5f7;
}
QLabel#opsLogo {
    background: transparent;
    padding: 4px 4px 12px 4px;
}
QPushButton#opsNavBtn {
    text-align: left;
    padding: 10px 12px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #e8edf3;
    font-size: 13px;
}
QPushButton#opsNavBtn:hover {
    background: #3a414d;
}
QPushButton#opsNavBtn:checked {
    background: #444c5a;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#opsNewBtn {
    text-align: left;
    padding: 8px 12px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #c5ced8;
    font-size: 12px;
}
QPushButton#opsNewBtn:hover {
    background: #3a414d;
    color: #ffffff;
}
QListWidget#opsTabList {
    background: #2c313a;
    border: none;
    color: #e8edf3;
    outline: none;
    font-size: 13px;
}
QListWidget#opsTabList::item {
    padding: 9px 12px;
    border-radius: 6px;
}
QListWidget#opsTabList::item:hover {
    background: #3a414d;
}
QListWidget#opsTabList::item:selected {
    background: #444c5a;
    color: #ffffff;
}
QLabel#opsUser {
    color: #b7c0cc;
    font-size: 12px;
    padding: 12px 8px 8px 8px;
}
"""


class OpsNavSidebar(QWidget):
    """Left nav: Home, telemetry tabs, Administration. AI Assistant lives on Home."""

    navigate = pyqtSignal(str)  # home | admin | new_tab | tab:<id>

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("opsNav")
        self.setFixedWidth(240)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#2c313a"))
        pal.setColor(QPalette.Base, QColor("#2c313a"))
        pal.setColor(QPalette.Text, QColor("#f3f5f7"))
        pal.setColor(QPalette.Button, QColor("#2c313a"))
        pal.setColor(QPalette.ButtonText, QColor("#e8edf3"))
        self.setPalette(pal)
        self.setStyleSheet(NAV_QSS)
        self._syncing = False
        self._build()

    def _nav_btn(self, text: str, key: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("opsNavBtn")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFlat(True)
        btn.clicked.connect(lambda _=False, k=key: self._emit(k))
        return btn

    def _build(self) -> None:
        inner = QFrame(self)
        inner.setObjectName("opsNavInner")
        inner.setAttribute(Qt.WA_StyledBackground, True)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(inner)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(2)

        logo = QLabel()
        logo.setObjectName("opsLogo")
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        pix = QPixmap(str(_LOGO_PATH))
        if not pix.isNull():
            logo.setPixmap(
                pix.scaled(200, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        layout.addWidget(logo)

        self.home_btn = self._nav_btn("Home", "home")
        layout.addWidget(self.home_btn)

        self.tab_list = QListWidget()
        self.tab_list.setObjectName("opsTabList")
        self.tab_list.setSpacing(1)
        self.tab_list.setFrameShape(QFrame.NoFrame)
        self.tab_list.itemClicked.connect(self._on_tab_item)
        layout.addWidget(self.tab_list, 1)

        self.new_tab_btn = QPushButton("+  New tab")
        self.new_tab_btn.setObjectName("opsNewBtn")
        self.new_tab_btn.setCursor(Qt.PointingHandCursor)
        self.new_tab_btn.setFlat(True)
        self.new_tab_btn.clicked.connect(lambda: self._emit("new_tab"))
        layout.addWidget(self.new_tab_btn)

        self.admin_btn = self._nav_btn("Administration", "admin")
        layout.addWidget(self.admin_btn)

        self.user_label = QLabel("")
        self.user_label.setObjectName("opsUser")
        self.user_label.setWordWrap(True)
        layout.addWidget(self.user_label)

    def _emit(self, key: str) -> None:
        if self._syncing:
            return
        self.navigate.emit(key)

    def _on_tab_item(self, item: QListWidgetItem) -> None:
        tab_id = item.data(Qt.UserRole)
        if tab_id:
            self._emit(f"tab:{tab_id}")

    def set_user(self, username: str, role: str) -> None:
        self.user_label.setText(f"{username}  ·  {role}")

    def set_can_create_tab(self, allowed: bool) -> None:
        self.new_tab_btn.setVisible(bool(allowed))

    def set_admin_visible(self, visible: bool) -> None:
        self.admin_btn.setVisible(bool(visible))

    def set_telemetry_tabs(self, tabs: list[tuple[str, str]]) -> None:
        """tabs: (tab_id, title)"""
        self._syncing = True
        current = None
        item = self.tab_list.currentItem()
        if item is not None:
            current = item.data(Qt.UserRole)
        self.tab_list.clear()
        for tab_id, title in tabs:
            row = QListWidgetItem(title or tab_id)
            row.setData(Qt.UserRole, tab_id)
            self.tab_list.addItem(row)
            if tab_id == current:
                self.tab_list.setCurrentItem(row)
        self._syncing = False

    def set_active(self, key: str) -> None:
        self._syncing = True
        # AI Assistant is a Home sub-page — keep Home highlighted there too.
        self.home_btn.setChecked(key in ("home", "ai"))
        self.admin_btn.setChecked(key == "admin")
        if not str(key).startswith("tab:"):
            self.tab_list.clearSelection()
        else:
            self.home_btn.setChecked(False)
            want = key.split(":", 1)[-1]
            for i in range(self.tab_list.count()):
                it = self.tab_list.item(i)
                if it.data(Qt.UserRole) == want:
                    self.tab_list.setCurrentItem(it)
                    break
        self._syncing = False


class ActiveTabsSidebar(QWidget):
    """Legacy list (unused by the ops shell). Kept for import compatibility."""

    tab_selected = None

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(QLabel("Active Tabs"))
        layout.addWidget(self.list_widget, 1)

    def set_tabs(self, tabs: list[tuple[str, str, str]]) -> None:
        self.list_widget.clear()
        for tab_id, title, status in tabs:
            item = QListWidgetItem(f"{title}\n{status}")
            item.setData(256, tab_id)
            self.list_widget.addItem(item)

    def current_tab_id(self) -> str | None:
        item = self.list_widget.currentItem()
        if not item:
            return None
        return item.data(256)
