from __future__ import annotations

import logging
from pathlib import Path

from app.config.settings import SettingsManager
from app.monitoring.scheduler import SCHEDULE_CONTINUOUS, SCHEDULE_ON_DEMAND, SCHEDULE_SCHEDULED, should_run_scheduled_tab
from app.monitoring.tab_config import TabConfigurationManager
from app.monitoring.tab_runtime import TabRuntimeState
from app.services.service_layer import AppServiceLayer, ServiceLayerContext
from app.ui.config_dialog import TabConfigDialog
from app.ui.overview import OverviewDashboard
from app.ui.qt_compat import (
    LeftDockWidgetArea,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTimer,
    QToolBar,
    QWidget,
    pyqtSignal,
)
from app.ui.sidebar import ActiveTabsSidebar
from app.ui.tab_view import MonitoringTabView
from app.ui.theme import ThemeManager

logger = logging.getLogger("STDMS.UI")


class MainWindow(QMainWindow):
    request_overview = pyqtSignal()

    def __init__(self, user_manager, username: str, role: str, workspace_root: Path):
        super().__init__()
        self.user_manager = user_manager
        self.username = username
        self.role = role
        self.workspace_root = workspace_root

        settings = SettingsManager(str(workspace_root)).load()
        self.data_dir = (workspace_root / settings.paths.data_dir) if not Path(settings.paths.data_dir).is_absolute() else Path(settings.paths.data_dir)
        self.reports_dir = (workspace_root / settings.paths.reports_dir) if not Path(settings.paths.reports_dir).is_absolute() else Path(settings.paths.reports_dir)
        self.tab_config_manager = TabConfigurationManager(self.data_dir / "custom_tabs_config.json")
        self.service_layer = AppServiceLayer(user_manager=user_manager)
        self.theme_manager = ThemeManager()

        self.runtimes: dict[str, TabRuntimeState] = {}
        self.tab_views: dict[str, MonitoringTabView] = {}
        self.tab_timers: dict[str, QTimer] = {}
        self.scheduled_last_run: dict[str, str | None] = {}

        self._build_ui()
        self._load_existing_tabs()
        self.show_overview()

    def _build_ui(self) -> None:
        self.setWindowTitle("SDA v4.0")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 820)

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        overview_btn = QPushButton("Overview")
        overview_btn.clicked.connect(self.show_overview)
        toolbar.addWidget(overview_btn)

        new_tab_btn = QPushButton("+ New Tab")
        new_tab_btn.setObjectName("primaryBtn")
        new_tab_btn.clicked.connect(self.create_tab)
        toolbar.addWidget(new_tab_btn)

        theme_btn = QPushButton("Toggle Theme")
        theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(theme_btn)

        spacer = QWidget()
        spacer.setMinimumWidth(20)
        toolbar.addWidget(spacer)

        user_label = QPushButton(f"{self.username} ({self.role})")
        user_label.setEnabled(False)
        toolbar.addWidget(user_label)

        self.sidebar = ActiveTabsSidebar()
        self.sidebar.list_widget.currentItemChanged.connect(self._on_sidebar_selection)
        dock = QDockWidget("Active Tabs", self)
        dock.setWidget(self.sidebar)
        dock.setMinimumWidth(240)
        self.addDockWidget(LeftDockWidgetArea, dock)

        self.stack = QStackedWidget()
        self.overview = OverviewDashboard()
        self.overview.refresh_btn.clicked.connect(self.refresh_overview)
        self.stack.addWidget(self.overview)
        self.setCentralWidget(self.stack)

        self.statusBar().showMessage(f"Offline mode | Logged in as {self.username} ({self.role})")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_overview)
        self.refresh_timer.start(5000)

    def _toggle_theme(self) -> None:
        from app.ui.qt_compat import QApplication

        app = QApplication.instance()
        if app:
            self.theme_manager.toggle(app)

    def _load_existing_tabs(self) -> None:
        for tab_id, config in self.tab_config_manager.list_tabs():
            self._register_tab(tab_id, config, select=False)
        self._refresh_sidebar()

    def create_tab(self) -> None:
        if not self.service_layer.authorize(self.username, "create_tab"):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to create tabs.")
            return
        config = TabConfigDialog.get_config(self)
        if not config:
            return
        tab_id = self.tab_config_manager.create_tab(config)
        self._register_tab(tab_id, config, select=True)
        self._refresh_sidebar()
        self.open_tab(tab_id)

    def edit_tab(self, tab_id: str) -> None:
        config = self.tab_config_manager.get_config(tab_id)
        if not config:
            return
        updated = TabConfigDialog.get_config(self, config)
        if not updated:
            return
        self.tab_config_manager.update_tab(tab_id, updated)
        runtime = self.runtimes[tab_id]
        runtime.config = updated
        runtime.snapshot.title = updated.get("title", "")
        if tab_id in self.tab_views:
            self.tab_views[tab_id].runtime = runtime
            self.tab_views[tab_id].title_label.setText(updated.get("title", ""))
        self._restart_timer(tab_id)
        self.refresh_overview()

    def _register_tab(self, tab_id: str, config: dict, select: bool = True) -> None:
        runtime = TabRuntimeState(tab_id=tab_id, config=config, data_dir=self.data_dir)
        self.runtimes[tab_id] = runtime
        view = MonitoringTabView(runtime, self.reports_dir, self)
        view.config_btn.clicked.connect(lambda _=False, tid=tab_id: self.edit_tab(tid))
        view.reload_btn.clicked.connect(lambda _=False, tid=tab_id: self._reload_tab(tid))
        view.train_btn.clicked.connect(lambda _=False, tid=tab_id: self._train_tab(tid))
        view.start_btn.clicked.connect(lambda _=False, tid=tab_id: self.start_monitoring(tid))
        view.stop_btn.clicked.connect(lambda _=False, tid=tab_id: self.stop_monitoring(tid))
        view.stop_btn.setEnabled(False)
        self.tab_views[tab_id] = view
        self.stack.addWidget(view)
        self._restart_timer(tab_id)
        if select:
            self.open_tab(tab_id)

    def _restart_timer(self, tab_id: str) -> None:
        if tab_id in self.tab_timers:
            self.tab_timers[tab_id].stop()
            self.tab_timers[tab_id].deleteLater()
        config = self.runtimes[tab_id].config
        schedule_type = config.get("schedule_type", SCHEDULE_CONTINUOUS)
        timer = QTimer(self)
        timer.timeout.connect(lambda tid=tab_id: self._tick_tab(tid))

        if schedule_type == SCHEDULE_CONTINUOUS:
            interval = max(1000, int(config.get("interval_ms", 300_000)))
            timer.start(interval)
        elif schedule_type == SCHEDULE_SCHEDULED:
            timer.start(30_000)
        else:
            pass
        self.tab_timers[tab_id] = timer

    def _tick_tab(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        if not runtime:
            return
        config = runtime.config
        schedule_type = config.get("schedule_type", SCHEDULE_CONTINUOUS)

        if schedule_type == SCHEDULE_ON_DEMAND:
            return

        if schedule_type == SCHEDULE_SCHEDULED:
            last_key = self.scheduled_last_run.get(tab_id)
            if not should_run_scheduled_tab(config, last_key):
                return
            hour = int(config.get("schedule_utc_hour", 0))
            minute = int(config.get("schedule_utc_minute", 0))
            from datetime import datetime

            self.scheduled_last_run[tab_id] = f"{datetime.utcnow().date().isoformat()}T{hour:02d}:{minute:02d}"

        if not runtime.monitoring_active and schedule_type == SCHEDULE_CONTINUOUS:
            return

        self._run_monitoring_cycle(tab_id)

    def start_monitoring(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        if not runtime:
            return
        runtime.set_monitoring(True)
        if tab_id in self.tab_views:
            self.tab_views[tab_id].set_monitoring_active(True)
        self._run_monitoring_cycle(tab_id)
        self.refresh_overview()

    def stop_monitoring(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        if not runtime:
            return
        runtime.set_monitoring(False)
        if tab_id in self.tab_views:
            self.tab_views[tab_id].set_monitoring_active(False)
        self.refresh_overview()

    def _run_monitoring_cycle(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        view = self.tab_views.get(tab_id)
        if not runtime or not view:
            return
        result = runtime.run_monitoring_cycle()
        if not result.get("ok") and result.get("error"):
            logger.info("Tab %s cycle: %s", tab_id, result.get("error"))
        view.refresh_view(result)
        self.refresh_overview()

    def _reload_tab(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        view = self.tab_views.get(tab_id)
        if not runtime or not view:
            return
        ok, err = runtime.reload_data(force=True)
        if not ok:
            QMessageBox.warning(self, "Data Load Error", err or "Failed to load data.")
        view.refresh_view(None)
        self.refresh_overview()

    def _train_tab(self, tab_id: str) -> None:
        runtime = self.runtimes.get(tab_id)
        view = self.tab_views.get(tab_id)
        if not runtime or not view:
            return
        if not self.service_layer.authorize(self.username, "train_models"):
            QMessageBox.warning(self, "Permission Denied", "You do not have permission to train models.")
            return
        ok, msg = runtime.train_models()
        if ok:
            QMessageBox.information(self, "Training Complete", msg)
        else:
            QMessageBox.warning(self, "Training Failed", msg)
        view.refresh_view(None)
        self.refresh_overview()

    def show_overview(self) -> None:
        self.stack.setCurrentWidget(self.overview)
        self.refresh_overview()

    def open_tab(self, tab_id: str) -> None:
        view = self.tab_views.get(tab_id)
        if not view:
            return
        self.stack.setCurrentWidget(view)
        for i in range(self.sidebar.list_widget.count()):
            item = self.sidebar.list_widget.item(i)
            if item and item.data(256) == tab_id:
                self.sidebar.list_widget.setCurrentItem(item)
                break

    def _on_sidebar_selection(self, current, _previous) -> None:
        if not current:
            return
        tab_id = current.data(256)
        if tab_id:
            self.open_tab(tab_id)

    def _refresh_sidebar(self) -> None:
        tabs = []
        for tab_id, runtime in self.runtimes.items():
            snap = runtime.get_snapshot()
            status = snap.get("watch_status", "Idle")
            if runtime.monitoring_active:
                status = f"● {status}"
            tabs.append((tab_id, runtime.config.get("title", tab_id), status))
        self.sidebar.set_tabs(tabs)

    def refresh_overview(self) -> None:
        snapshots = [runtime.get_snapshot() for runtime in self.runtimes.values()]
        self.overview.update_snapshots(snapshots)
        self._refresh_sidebar()
