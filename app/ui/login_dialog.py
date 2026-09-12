from __future__ import annotations

from pathlib import Path

from app.ui.qt_compat import (
    AlignCenter,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPixmap,
    QPushButton,
    QVBoxLayout,
    Qt,
    exec_dialog,
)

AZERCOSMOS_LOGO = Path(__file__).resolve().parent / "assets" / "azercosmos-logo.png"


def load_azercosmos_logo(*, width: int = 360, height: int = 90) -> QPixmap:
    pix = QPixmap(str(AZERCOSMOS_LOGO))
    if pix.isNull():
        return pix
    return pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class LoginDialog(QDialog):
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.username: str | None = None
        self.role: str | None = None
        self.password_change_required = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("SDA v4.0 — Sign In")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        logo = QLabel()
        logo.setAlignment(AlignCenter)
        pix = load_azercosmos_logo()
        if not pix.isNull():
            logo.setPixmap(pix)
        else:
            logo.setText("Azercosmos")
            logo.setStyleSheet("font-size: 18px; font-weight: 700; color: #2f6fad;")
        layout.addWidget(logo)

        header = QLabel("SDA v4.0")
        header.setWordWrap(True)
        header.setAlignment(AlignCenter)
        header.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(header)

        sub = QLabel("Local offline monitoring — sign in to continue")
        sub.setStyleSheet("color: #5c6478;")
        layout.addWidget(sub)

        form = QFormLayout()
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        echo_mode = QLineEdit.EchoMode.Password if hasattr(QLineEdit, "EchoMode") else 2
        self.password_input.setEchoMode(echo_mode)
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)
        layout.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c53030;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        login_btn = QPushButton("Sign In")
        login_btn.setObjectName("primaryBtn")
        login_btn.clicked.connect(self._attempt_login)
        login_btn.setDefault(True)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(login_btn)
        layout.addLayout(buttons)

    def _attempt_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if not username or not password:
            self.error_label.setText("Enter username and password.")
            return

        ok, role, message, change_required = self.user_manager.authenticate(username, password)
        if not ok:
            self.error_label.setText(message)
            return

        self.username = username
        self.role = role
        self.password_change_required = change_required
        self.accept()

    @staticmethod
    def run(user_manager, parent=None) -> tuple[bool, str | None, str | None]:
        dialog = LoginDialog(user_manager, parent)
        accepted_code = QDialog.DialogCode.Accepted if hasattr(QDialog, "DialogCode") else QDialog.Accepted
        if exec_dialog(dialog) != accepted_code:
            return False, None, None
        return True, dialog.username, dialog.role
