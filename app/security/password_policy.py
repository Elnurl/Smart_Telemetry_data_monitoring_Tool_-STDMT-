from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PasswordPolicy:
    min_length: int = 12
    require_upper: bool = True
    require_lower: bool = True
    require_digit: bool = True
    require_special: bool = True

    def validate(self, password: str) -> Tuple[bool, str]:
        if not password or len(password) < self.min_length:
            return False, f"Password must be at least {self.min_length} characters."
        if self.require_upper and not any(ch.isupper() for ch in password):
            return False, "Password must include at least one uppercase letter."
        if self.require_lower and not any(ch.islower() for ch in password):
            return False, "Password must include at least one lowercase letter."
        if self.require_digit and not any(ch.isdigit() for ch in password):
            return False, "Password must include at least one digit."
        if self.require_special and not any(not ch.isalnum() for ch in password):
            return False, "Password must include at least one special character."
        return True, "Password policy satisfied."

