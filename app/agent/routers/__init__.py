"""Versioned ToolHost HTTP routers (JWT operator API)."""

from __future__ import annotations

import hmac
from contextlib import nullcontext
from typing import Any, Optional

from fastapi import Header, HTTPException, Request

from app.agent.bridge import get_tool_host
from app.agent.jwt_tokens import decode_access_token


def attached_host():
    host = get_tool_host()
    if host is None:
        raise HTTPException(status_code=503, detail="Tool host not attached")
    return host


def operator_scope(host: Any, identity: Optional[dict[str, Any]]):
    as_op = getattr(host, "as_operator", None)
    username = str((identity or {}).get("username") or "api")
    if callable(as_op):
        return as_op(username)
    return nullcontext()


def raise_from_host(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("ok"):
        return result
    status = str(result.get("status") or "")
    if status in {"tab_not_found", "model_not_found", "not_found"}:
        code = 404
    elif status == "forbidden":
        code = 403
    elif status in {"invalid_config", "not_ready"}:
        code = 400
    else:
        code = 409
    raise HTTPException(status_code=code, detail=result)


def _bearer_token(authorization: Optional[str]) -> str:
    scheme, _, supplied = (authorization or "").partition(" ")
    token = supplied.strip()
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def require_v1_auth(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """Accept a JWT access token, or the legacy static API bearer token."""
    token = _bearer_token(authorization)
    blacklist = getattr(request.app.state, "token_blacklist", set())
    if token in blacklist:
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    static = str(getattr(request.app.state, "api_token", "") or "")
    if static and hmac.compare_digest(token, static):
        return {
            "username": "api",
            "role": "admin",
            "jti": None,
            "token": token,
            "kind": "static",
        }
    secret = str(getattr(request.app.state, "jwt_secret", "") or "")
    try:
        payload = decode_access_token(token, secret)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    jti = str(payload.get("jti") or "")
    if jti and jti in blacklist:
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "username": str(payload.get("sub") or ""),
        "role": str(payload.get("role") or "viewer"),
        "jti": jti,
        "token": token,
        "kind": "jwt",
    }
