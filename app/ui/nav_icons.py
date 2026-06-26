"""Solid monochrome navigation icons (Yamcs-style, no emoji / no system pixmap)."""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget

_ICON_COLOR = QColor("#3a424f")
_ICON_SIZE = 20

# (label, icon_key)
CUSTOM_TAB_NAV_ENTRIES = (
    ("Quick Actions", "quick_actions"),
    ("Model Development", "model_dev"),
    ("Analysis / ML", "analysis"),
    ("Data Import", "data_import"),
    ("Visualization", "visualization"),
)


def _draw_quick_actions(p: QPainter, size: int) -> None:
    m = size * 0.22
    points = [
        (m, m),
        (size - m, size * 0.5),
        (m, size - m),
    ]
    from PyQt5.QtGui import QPolygonF
    from PyQt5.QtCore import QPointF

    p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))


def _draw_model_dev(p: QPainter, size: int) -> None:
    m = size * 0.18
    w = size - 2 * m
    h = size - 2 * m
    p.drawRoundedRect(int(m), int(m), int(w), int(h), 2, 2)
    line_h = max(2, int(size * 0.12))
    gap = size * 0.16
    y = m + gap
    for _ in range(3):
        p.drawRect(int(m + gap * 0.6), int(y), int(w - gap * 1.2), line_h)
        y += gap


def _draw_analysis(p: QPainter, size: int) -> None:
    m = size * 0.16
    screen_h = size * 0.52
    p.drawRoundedRect(int(m), int(m), int(size - 2 * m), int(screen_h), 2, 2)
    base_w = size * 0.34
    base_h = size * 0.1
    p.drawRoundedRect(
        int((size - base_w) / 2),
        int(m + screen_h + size * 0.06),
        int(base_w),
        int(base_h),
        1,
        1,
    )


def _draw_data_import(p: QPainter, size: int) -> None:
    m = size * 0.18
    tray_h = size * 0.14
    p.drawRoundedRect(int(m), int(size * 0.56), int(size - 2 * m), int(tray_h), 2, 2)
    shaft_w = size * 0.14
    shaft_x = (size - shaft_w) / 2
    p.drawRect(int(shaft_x), int(size * 0.28), int(shaft_w), int(size * 0.3))
    head = size * 0.28
    from PyQt5.QtGui import QPolygonF
    from PyQt5.QtCore import QPointF

    cx = size / 2
    p.drawPolygon(
        QPolygonF(
            [
                QPointF(cx - head / 2, size * 0.42),
                QPointF(cx + head / 2, size * 0.42),
                QPointF(cx, size * 0.58),
            ]
        )
    )


def _draw_visualization(p: QPainter, size: int) -> None:
    m = size * 0.18
    gap = size * 0.1
    cell = (size - 2 * m - gap) / 2
    for row in range(2):
        for col in range(2):
            x = m + col * (cell + gap)
            y = m + row * (cell + gap)
            p.drawRoundedRect(int(x), int(y), int(cell), int(cell), 2, 2)


_DRAWERS: dict[str, Callable[[QPainter, int], None]] = {
    "quick_actions": _draw_quick_actions,
    "model_dev": _draw_model_dev,
    "analysis": _draw_analysis,
    "data_import": _draw_data_import,
    "visualization": _draw_visualization,
}


def solid_nav_icon(key: str, size: int = _ICON_SIZE, color: QColor | None = None) -> QIcon:
    """Return a filled monochrome icon for sidebar navigation."""
    drawer = _DRAWERS.get(key)
    if drawer is None:
        return QIcon()

    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(color or _ICON_COLOR)
    drawer(painter, size)
    painter.end()
    return QIcon(pm)


def nav_icon(widget: QWidget, key: str) -> QIcon:
    """Sidebar icon factory (widget kept for API compatibility)."""
    _ = widget
    return solid_nav_icon(key)


def nav_icon_size() -> QSize:
    return QSize(_ICON_SIZE, _ICON_SIZE)
