from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path


class SecurityUtils:
    @staticmethod
    def hash_password(password: str, salt: bytes | None = None) -> bytes:
        if salt is None:
            salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return salt + key

    @staticmethod
    def verify_password(stored_password: bytes, provided_password: str) -> bool:
        salt = stored_password[:32]
        key = stored_password[32:]
        new_key = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, 100_000)
        return new_key == key

    @staticmethod
    def load_encryption_key(key_file: Path) -> bytes:
        key_file.parent.mkdir(parents=True, exist_ok=True)
        if key_file.exists():
            return key_file.read_bytes()
        key = secrets.token_bytes(32)
        key_file.write_bytes(key)
        return key

    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        iv = secrets.token_bytes(16)
        stretched_key = b""
        for i in range(0, len(data), len(key)):
            stretched_key += hashlib.sha256(key + str(i).encode()).digest()
        stretched_key = stretched_key[: len(data)]
        encrypted = bytes(data[i] ^ stretched_key[i] for i in range(len(data)))
        return base64.b64encode(iv + encrypted)

    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
        raw = base64.b64decode(encrypted_data)
        data = raw[16:]
        stretched_key = b""
        for i in range(0, len(data), len(key)):
            stretched_key += hashlib.sha256(key + str(i).encode()).digest()
        stretched_key = stretched_key[: len(data)]
        return bytes(data[i] ^ stretched_key[i] for i in range(len(data)))
