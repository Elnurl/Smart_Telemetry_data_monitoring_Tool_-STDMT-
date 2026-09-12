"""HS256 JWT helpers for the local ToolHost API (loopback only)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Optional

JWT_TTL_SECONDS = 8 * 60 * 60
JWT_ALG = "HS256"


def resolve_jwt_secret(explicit: Optional[str] = None) -> str:
    secret = (explicit or os.getenv("STDMS_JWT_SECRET") or "").strip()
    if secret:
        return secret
    return secrets.token_urlsafe(48)


def issue_access_token(
    *,
    username: str,
    role: str,
    secret: str,
    ttl_seconds: int = JWT_TTL_SECONDS,
) -> str:
    now = int(time.time())
    payload = {
        "sub": str(username),
        "role": str(role or "viewer"),
        "iat": now,
        "exp": now + max(60, int(ttl_seconds)),
        "jti": secrets.token_urlsafe(12),
    }
    try:
        import jwt

        return jwt.encode(payload, secret, algorithm=JWT_ALG)
    except Exception:
        body = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"stdms.{body}.{sig}"


def _decode_fallback(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "stdms":
        raise ValueError("invalid_token")
    _hdr, body, sig = parts
    expected = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid_token")
    pad = "=" * (-len(body) % 4)
    payload = json.loads(base64.urlsafe_b64decode(body + pad).decode("utf-8"))
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("expired_token")
    return payload


def decode_access_token(token: str, secret: str) -> dict[str, Any]:
    raw = (token or "").strip()
    if not raw:
        raise ValueError("empty_token")
    if raw.startswith("stdms."):
        return _decode_fallback(raw, secret)
    try:
        import jwt
    except ImportError as exc:
        raise ValueError("invalid_token") from exc
    try:
        return jwt.decode(raw, secret, algorithms=[JWT_ALG])
    except Exception as exc:
        raise ValueError("invalid_token") from exc
