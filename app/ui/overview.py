from __future__ import annotations

from app.ui.qt_compat import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "0", parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        layout = QVBoxLayout(self)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statValue")
        self.text_label = QLabel(label)
        self.text_label.setObjectName("statLabel")
        layout.addWidget(self.value_label)
        layout.addWidget(self.text_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class OverviewDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        header = QLabel("Fleet Overview")
        header.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(header)

        sub = QLabel("Real-time status of all monitoring tabs — offline local operation")
        sub.setStyleSheet("color: #5c6478;")
        layout.addWidget(sub)

        cards = QHBoxLayout()
        self.total_tabs = StatCard("Total Tabs")
        self.active_tabs = StatCard("Monitoring Active")
        self.alert_tabs = StatCard("Watch Alerts")
        self.obs_tabs = StatCard("OBS Violations")
        for card in (self.total_tabs, self.active_tabs, self.alert_tabs, self.obs_tabs):
            cards.addWidget(card)
        layout.addLayout(cards)

        fleet_group = QGroupBox("Tab Snapshots")
        fleet_layout = QVBoxLayout(fleet_group)
        self.fleet_table = QTableWidget(0, 9)
        self.fleet_table.setHorizontalHeaderLabels(
            ["Tab", "Monitoring", "Health", "Watch", "Last File", "Rows", "OBS", "Drift", "Updated"]
        )
        self.fleet_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch if hasattr(QHeaderView, "ResizeMode") else QHeaderView.Stretch)
        self.fleet_table.setAlternatingRowColors(True)
        self.fleet_table.setMinimumHeight(260)
        fleet_layout.addWidget(self.fleet_table)

        self.refresh_btn = QPushButton("Refresh Overview")
        fleet_layout.addWidget(self.refresh_btn)
        layout.addWidget(fleet_group, 1)

    def update_snapshots(self, snapshots: list[dict]) -> None:
        total = len(snapshots)
        active = sum(1 for s in snapshots if s.get("monitoring_active"))
        alerts = sum(1 for s in snapshots if not s.get("obs_ok", True) or s.get("watch_status") not in ("OK", "Idle", ""))
        obs_bad = sum(1 for s in snapshots if not s.get("obs_ok", True))

        self.total_tabs.set_value(str(total))
        self.active_tabs.set_value(str(active))
        self.alert_tabs.set_value(str(alerts))
        self.obs_tabs.set_value(str(obs_bad))

        self.fleet_table.setRowCount(total)
        for row, snap in enumerate(snapshots):
            last_file = str(snap.get("last_file", "—"))
            if len(last_file) > 48:
                last_file = "…" + last_file[-45:]
            values = [
                snap.get("title", "—"),
                "Yes" if snap.get("monitoring_active") else "No",
                snap.get("health_state", "Idle"),
                snap.get("watch_status", "—"),
                last_file,
                str(snap.get("last_file_rows", 0)),
                "OK" if snap.get("obs_ok", True) else f"{snap.get('obs_violations', 0)} viol.",
                "Yes" if snap.get("drift") else "No",
                snap.get("updated_at", "—"),
            ]
            for col, text in enumerate(values):
                self.fleet_table.setItem(row, col, QTableWidgetItem(text))
