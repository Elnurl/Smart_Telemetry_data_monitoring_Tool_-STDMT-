#!/usr/bin/env python3
"""STDMS v2 — New modular desktop entry point (local offline)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("STDMS")

    from app.config.settings import SettingsManager
    from app.auth.user_manager import UserManager
    from app.ui.qt_compat import QApplication, QMessageBox, exec_app
    from app.ui.theme import ThemeManager
    from app.ui.login_dialog import LoginDialog
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("STDMS")
    app.setOrganizationName("Satellite Telemetry Monitoring")

    ThemeManager().apply(app)

    settings_mgr = SettingsManager(str(ROOT))
    settings = settings_mgr.load()
    user_manager = UserManager(
        user_db_file=ROOT / settings.paths.user_db_file,
        encryption_key_file=ROOT / settings.paths.encryption_key_file,
    )

    ok, username, role = LoginDialog.run(user_manager)
    if not ok or not username:
        logger.info("Login cancelled")
        return 0

    try:
        window = MainWindow(user_manager, username, role or "viewer", ROOT)
        window.show()
    except Exception as exc:
        logger.exception("Failed to start main window")
        QMessageBox.critical(None, "Startup Error", f"Failed to start STDMS:\n{exc}")
        return 1

    return exec_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
