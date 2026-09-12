"""PyQt5-first compatibility layer with PyQt6 fallback.

Matches run.bat / main.py (PyQt5). Modular app.ui code can still run on PyQt6
if PyQt5 is absent.
"""

from __future__ import annotations

try:
    from PyQt5.QtCore import (  # type: ignore
        QEasingCurve,
        QEvent,
        QObject,
        QPropertyAnimation,
        QSize,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt5.QtGui import (  # type: ignore
        QColor,
        QFont,
        QIcon,
        QKeyEvent,
        QPalette,
        QPixmap,
    )
    from PyQt5.QtWidgets import (  # type: ignore
        QAction,
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    QT_VERSION = 5

    def exec_dialog(dialog: QDialog) -> int:
        return dialog.exec_()

    def exec_app(app: QApplication) -> int:
        return app.exec_()

    ItemIsUserCheckable = Qt.ItemIsUserCheckable
    Checked = Qt.Checked
    Unchecked = Qt.Unchecked
    AlignCenter = Qt.AlignCenter
    AlignLeft = Qt.AlignLeft
    ScrollBarAsNeeded = Qt.ScrollBarAsNeeded
    ToolButtonTextOnly = Qt.ToolButtonTextOnly
    InstantPopup = QToolButton.InstantPopup
    LeftDockWidgetArea = Qt.LeftDockWidgetArea

except ImportError:
    from PyQt6.QtCore import (
        QEasingCurve,
        QEvent,
        QObject,
        QPropertyAnimation,
        QSize,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QAction,
        QColor,
        QFont,
        QIcon,
        QKeyEvent,
        QPalette,
        QPixmap,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QToolBar,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    QT_VERSION = 6

    def exec_dialog(dialog: QDialog) -> int:
        return dialog.exec()

    def exec_app(app: QApplication) -> int:
        return app.exec()

    ItemIsUserCheckable = Qt.ItemFlag.ItemIsUserCheckable
    Checked = Qt.CheckState.Checked
    Unchecked = Qt.CheckState.Unchecked
    AlignCenter = Qt.AlignmentFlag.AlignCenter
    AlignLeft = Qt.AlignmentFlag.AlignLeft
    ScrollBarAsNeeded = Qt.ScrollBarPolicy.ScrollBarAsNeeded
    ToolButtonTextOnly = Qt.ToolButtonStyle.ToolButtonTextOnly
    InstantPopup = QToolButton.ToolButtonPopupMode.InstantPopup
    LeftDockWidgetArea = (
        Qt.DockWidgetArea.LeftDockWidgetArea
        if hasattr(Qt, "DockWidgetArea")
        else Qt.LeftDockWidgetArea
    )
