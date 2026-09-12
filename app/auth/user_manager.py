from __future__ import annotations

import datetime
import json
import logging
import secrets
from pathlib import Path
from typing import Any

from app.auth.security import SecurityUtils
from app.security.password_policy import PasswordPolicy

logger = logging.getLogger("STDMS.Auth")


class UserManager:
    ROLES = {
        "admin": ["view_data", "import_data", "process_data", "manage_users", "configure_system", "train_models"],
        "analyst": ["view_data", "import_data", "process_data", "train_models"],
        "viewer": ["view_data", "import_data", "process_data"],
    }

    PERMISSION_MATRIX = {
        "view_data": {"*": ["admin", "analyst", "viewer"]},
        "import_data": {"*": ["admin", "analyst", "viewer"], "data_source:*": ["admin", "analyst"]},
        "process_data": {"*": ["admin", "analyst"]},
        "train_models": {"*": ["admin", "analyst"], "model:production": ["admin"]},
        "manage_users": {"*": ["admin"]},
        "configure_system": {"*": ["admin"]},
        "ack_alert": {"*": ["admin", "analyst"], "alert:*": ["admin", "analyst"]},
        "create_tab": {"*": ["admin", "analyst"], "tab:*": ["admin", "analyst"]},
        "delete_tab": {"tab:*": ["admin"], "*": ["admin"]},
    }

    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_MINUTES = 15
    MIN_PASSWORD_LENGTH = 12

    def __init__(self, user_db_file: Path, encryption_key_file: Path):
        self.user_db_file = user_db_file
        self.encryption_key_file = encryption_key_file
        self.encryption_key = SecurityUtils.load_encryption_key(encryption_key_file)
        self.users: dict[str, dict[str, Any]] = {}
        self.load_users()

    def _password_meets_policy(self, password: str) -> tuple[bool, str]:
        policy = PasswordPolicy(min_length=self.MIN_PASSWORD_LENGTH)
        return policy.validate(password)

    def _generate_bootstrap_password(self) -> str:
        return f"Tmp!{secrets.token_urlsafe(10)}A9"

    def _create_bootstrap_admin(self, reason: str = "initial setup") -> None:
        bootstrap_password = self._generate_bootstrap_password()
        self.users = {
            "admin": {
                "password": SecurityUtils.hash_password(bootstrap_password).hex(),
                "role": "admin",
                "email": "",
                "auth_provider": "local",
                "password_change_required": True,
                "failed_attempts": 0,
                "lockout_until": None,
                "last_password_change": None,
            }
        }
        self.save_users()
        bootstrap_file = self.user_db_file.parent / "bootstrap_admin_password.txt"
        bootstrap_file.write_text(
            "SDA v4.0 - Initial Admin Credentials\n"
            "---------------------------------\n"
            f"Reason: {reason}\n"
            "Username: admin\n"
            f"Temporary Password: {bootstrap_password}\n\n"
            "Change this password after first login.\n",
            encoding="utf-8",
        )
        logger.warning("Bootstrap admin credentials written to %s", bootstrap_file)

    def load_users(self) -> None:
        self.user_db_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.user_db_file.exists():
            self._create_bootstrap_admin(reason="first run")
            return
        try:
            encrypted = self.user_db_file.read_bytes()
            decrypted = SecurityUtils.decrypt_data(encrypted, self.encryption_key)
            self.users = json.loads(decrypted.decode("utf-8"))
            if not isinstance(self.users, dict):
                raise ValueError("Invalid user database format")
            changed = False
            for user in self.users.values():
                email = str(user.get("email", "")).strip()
                if email and ("@" not in email or email.split("@", 1)[-1].strip().lower() in {
                    "example.com", "example.org", "example.net"
                }):
                    user["email"] = ""
                    changed = True
            if changed:
                self.save_users()
        except Exception as exc:
            logger.error("Failed to load users: %s", exc)
            self._create_bootstrap_admin(reason="recovery from user database load failure")

    def save_users(self) -> None:
        payload = json.dumps(self.users).encode("utf-8")
        encrypted = SecurityUtils.encrypt_data(payload, self.encryption_key)
        self.user_db_file.write_bytes(encrypted)

    def check_permission(self, username: str, permission: str, resource: str = "*") -> bool:
        user = self.users.get(username)
        if not user:
            return False
        role = user.get("role")
        matrix = self.PERMISSION_MATRIX.get(permission, {})
        allowed = matrix.get(resource) or matrix.get("*") or []
        return role in allowed

    def _get_lockout_remaining_seconds(self, user: dict[str, Any]) -> int:
        lockout_until = user.get("lockout_until")
        if not lockout_until:
            return 0
        try:
            unlock_at = datetime.datetime.fromisoformat(lockout_until)
            remaining = (unlock_at - datetime.datetime.utcnow()).total_seconds()
            if remaining <= 0:
                user["lockout_until"] = None
                user["failed_attempts"] = 0
                self.save_users()
                return 0
            return int(remaining)
        except Exception:
            user["lockout_until"] = None
            user["failed_attempts"] = 0
            self.save_users()
            return 0

    def _authenticate_local(self, username: str, password: str) -> tuple[bool, str | None, str, bool]:
        """Local username/password check used by LocalAuthProvider and ToolHost login."""
        return self.authenticate(username, password)

    def authenticate(self, username: str, password: str) -> tuple[bool, str | None, str, bool]:
        """Returns success, role, message, password_change_required."""
        user = self.users.get(username)
        if not user:
            return False, None, "Invalid username or password.", False

        remaining = self._get_lockout_remaining_seconds(user)
        if remaining > 0:
            return False, None, f"Account locked. Try again in {remaining // 60 + 1} minute(s).", False

        try:
            stored = bytes.fromhex(user["password"])
        except Exception:
            return False, None, "Invalid username or password.", False

        if not SecurityUtils.verify_password(stored, password):
            user["failed_attempts"] = int(user.get("failed_attempts", 0)) + 1
            if user["failed_attempts"] >= self.MAX_FAILED_ATTEMPTS:
                unlock = datetime.datetime.utcnow() + datetime.timedelta(minutes=self.LOCKOUT_MINUTES)
                user["lockout_until"] = unlock.isoformat()
                self.save_users()
                return False, None, f"Too many failed attempts. Locked for {self.LOCKOUT_MINUTES} minutes.", False
            self.save_users()
            return False, None, "Invalid username or password.", False

        user["failed_attempts"] = 0
        user["lockout_until"] = None
        self.save_users()
        return True, user.get("role"), "Login successful.", bool(user.get("password_change_required"))
