from __future__ import annotations

from app.ui.qt_compat import QApplication, QPalette, QColor


LIGHT_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f5f7fb;
    color: #1a1d26;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QToolBar {
    background: #ffffff;
    border-bottom: 1px solid #dfe3eb;
    padding: 6px;
    spacing: 8px;
}
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
}
QDockWidget::title {
    background: #ffffff;
    padding: 8px;
    border-bottom: 1px solid #dfe3eb;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #c8ceda;
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:hover {
    background: #eef2ff;
    border-color: #4f6ef7;
}
QPushButton#primaryBtn {
    background: #4f6ef7;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background: #3f59d9;
}
QPushButton#dangerBtn {
    background: #fff5f5;
    color: #c53030;
    border-color: #feb2b2;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #dfe3eb;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #c8ceda;
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 28px;
}
QTableWidget {
    background: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: 8px;
    gridline-color: #edf0f5;
}
QHeaderView::section {
    background: #f0f3f9;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #dfe3eb;
    font-weight: 600;
}
QListWidget {
    background: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: 8px;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #dfe3eb;
}
#sidebarList::item {
    padding: 10px 12px;
    border-radius: 6px;
    margin: 2px 4px;
}
#sidebarList::item:selected {
    background: #4f6ef7;
    color: white;
}
#statCard {
    background: #ffffff;
    border: 1px solid #dfe3eb;
    border-radius: 10px;
    padding: 16px;
}
#statValue {
    font-size: 28px;
    font-weight: 700;
    color: #4f6ef7;
}
#statLabel {
    color: #5c6478;
}
"""

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #12151c;
    color: #e8ebf3;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QToolBar {
    background: #1a1f2b;
    border-bottom: 1px solid #2a3142;
    padding: 6px;
}
QDockWidget::title {
    background: #1a1f2b;
    padding: 8px;
    border-bottom: 1px solid #2a3142;
}
QPushButton {
    background: #1f2533;
    border: 1px solid #3a4258;
    border-radius: 6px;
    padding: 8px 14px;
    color: #e8ebf3;
}
QPushButton:hover {
    background: #2a3550;
    border-color: #6b8cff;
}
QPushButton#primaryBtn {
    background: #6b8cff;
    color: #0f1117;
    border: none;
    font-weight: 600;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #2a3142;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    background: #1a1f2b;
}
QLineEdit, QComboBox, QSpinBox {
    background: #12151c;
    border: 1px solid #3a4258;
    border-radius: 6px;
    padding: 6px 8px;
    color: #e8ebf3;
}
QTableWidget, QListWidget {
    background: #1a1f2b;
    border: 1px solid #2a3142;
    border-radius: 8px;
    gridline-color: #2a3142;
}
QHeaderView::section {
    background: #1f2533;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #2a3142;
}
QStatusBar {
    background: #1a1f2b;
    border-top: 1px solid #2a3142;
}
#sidebarList::item:selected {
    background: #6b8cff;
    color: #0f1117;
}
#statCard {
    background: #1a1f2b;
    border: 1px solid #2a3142;
    border-radius: 10px;
    padding: 16px;
}
#statValue {
    font-size: 28px;
    font-weight: 700;
    color: #6b8cff;
}
#statLabel {
    color: #9aa3b8;
}
"""


class ThemeManager:
    def __init__(self) -> None:
        self._dark = False

    @property
    def is_dark(self) -> bool:
        return self._dark

    def apply(self, app: QApplication) -> None:
        app.setStyle("Fusion")
        window_role = QPalette.ColorRole.Window if hasattr(QPalette, "ColorRole") else QPalette.Window
        text_role = QPalette.ColorRole.WindowText if hasattr(QPalette, "ColorRole") else QPalette.WindowText
        if self._dark:
            app.setStyleSheet(DARK_STYLESHEET)
            palette = QPalette()
            palette.setColor(window_role, QColor("#12151c"))
            palette.setColor(text_role, QColor("#e8ebf3"))
            app.setPalette(palette)
        else:
            app.setStyleSheet(LIGHT_STYLESHEET)
            palette = QPalette()
            palette.setColor(window_role, QColor("#f5f7fb"))
            palette.setColor(text_role, QColor("#1a1d26"))
            app.setPalette(palette)

    def toggle(self, app: QApplication) -> None:
        self._dark = not self._dark
        self.apply(app)
