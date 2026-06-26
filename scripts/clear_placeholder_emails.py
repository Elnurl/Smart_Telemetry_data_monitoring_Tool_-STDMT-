"""One-shot: remove placeholder example.com emails from encrypted users.json."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from cryptography.fernet import Fernet

PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    key_file = root / "keys" / "encryption_key.bin"
    users_file = root / "data" / "users.json"
    key = key_file.read_bytes()
    fernet = Fernet(base64.urlsafe_b64encode(key[:32]))
    raw = base64.b64decode(users_file.read_bytes())
    if not raw.startswith(b"F1:"):
        print("Unexpected users.json format", file=sys.stderr)
        return 1
    users = json.loads(fernet.decrypt(raw[3:]).decode("utf-8"))
    changed = []
    for username, info in users.items():
        email = str(info.get("email", "")).strip()
        if email and email.split("@", 1)[-1].strip().lower() in PLACEHOLDER_DOMAINS:
            info["email"] = ""
            changed.append(username)
    if changed:
        encrypted = base64.b64encode(b"F1:" + fernet.encrypt(json.dumps(users).encode("utf-8")))
        users_file.write_bytes(encrypted)
    print("cleared placeholder email for:", ", ".join(changed) if changed else "(none)")
    for username, info in users.items():
        print(f"  {username}: {info.get('email', '')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
