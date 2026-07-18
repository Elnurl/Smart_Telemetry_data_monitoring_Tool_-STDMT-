"""Capture Keras/training stdout and mirror it into a Qt Training Console."""

from __future__ import annotations

import contextlib
import sys
from typing import Any, Iterator, Optional, TextIO

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QPlainTextEdit, QTextEdit


class TrainingConsoleBridge(QObject):
    """Thread-safe bridge: worker emits chunks, UI appends on the Qt thread."""

    chunk = pyqtSignal(str)


class StdoutCapture:
    """Text IO that tees writes to the real stdout and a TrainingConsoleBridge."""

    def __init__(self, original: TextIO, bridge: TrainingConsoleBridge):
        self._original = original
        self._bridge = bridge

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._original.write(data)
            self._original.flush()
        except Exception:
            pass
        try:
            self._bridge.chunk.emit(data)
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self._original, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._original.fileno()


@contextlib.contextmanager
def capture_stdout(bridge: TrainingConsoleBridge) -> Iterator[None]:
    """Temporarily redirect sys.stdout to tee into the bridge."""
    original = sys.stdout
    sys.stdout = StdoutCapture(original, bridge)
    try:
        yield
    finally:
        sys.stdout = original


def append_console_chunk(widget: Any, text: str) -> None:
    """Append training output; treat ``\\r`` as overwrite of the current line (Keras bars)."""
    if widget is None or not text:
        return

    # Prefer QPlainTextEdit / QTextEdit APIs
    is_plain = isinstance(widget, QPlainTextEdit)
    is_rich = isinstance(widget, QTextEdit) and not is_plain
    if not is_plain and not is_rich and not hasattr(widget, "append"):
        return

    # Process carriage returns: split on \r and keep only the last segment per "line update"
    parts = text.split("\r")
    if len(parts) == 1:
        _append_text(widget, text, is_plain=is_plain, is_rich=is_rich)
        return

    # First segment may continue the previous line; subsequent segments replace it
    for i, part in enumerate(parts):
        if i == 0:
            if part:
                _append_text(widget, part, is_plain=is_plain, is_rich=is_rich)
        else:
            _replace_last_line(widget, part, is_plain=is_plain, is_rich=is_rich)


def _append_text(widget: Any, text: str, *, is_plain: bool, is_rich: bool) -> None:
    if is_plain or is_rich:
        cursor = widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        widget.setTextCursor(cursor)
        widget.insertPlainText(text)
        widget.ensureCursorVisible()
    else:
        # QTextEdit-style append expects full lines; fall back carefully
        if text.endswith("\n"):
            widget.append(text.rstrip("\n"))
        else:
            cursor = widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            widget.setTextCursor(cursor)
            widget.insertPlainText(text)


def _replace_last_line(widget: Any, text: str, *, is_plain: bool, is_rich: bool) -> None:
    if not (is_plain or is_rich):
        if text:
            widget.append(text.rstrip("\n"))
        return

    cursor = widget.textCursor()
    cursor.movePosition(QTextCursor.End)
    cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
    cursor.removeSelectedText()
    cursor.insertText(text)
    widget.setTextCursor(cursor)
    widget.ensureCursorVisible()


def clear_training_console(host: Any) -> None:
    console = getattr(host, "training_console", None)
    if console is not None and hasattr(console, "clear"):
        console.clear()


def append_training_header(host: Any, message: str) -> None:
    console = getattr(host, "training_console", None)
    if console is None:
        return
    append_console_chunk(console, f"\n=== {message} ===\n")
