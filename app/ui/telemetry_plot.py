from __future__ import annotations

import matplotlib

from app.ui.qt_compat import QT_VERSION, QVBoxLayout, QWidget

if QT_VERSION >= 6:
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
else:
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure


class TelemetryPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(6, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot_series(self, x, y, feature: str) -> None:
        ax = self.figure.gca()
        ax.clear()
        ax.plot(x, y, linewidth=1.4, color="#4f6ef7")
        ax.set_title(f"Telemetry — {feature}")
        ax.set_xlabel("Time" if hasattr(x, "__iter__") and not isinstance(x, range) else "Index")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.figure.clf()
        self.canvas.draw_idle()
