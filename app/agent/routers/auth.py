"""POST /v1/auth/login, logout, and current-user identity."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.jwt_tokens import issue_access_token
from app.agent.routers import require_v1_auth

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=500)


def _user_manager(request: Request):
    host = None
    try:
        from app.agent.bridge import get_tool_host

        host = get_tool_host()
    except Exception:
        host = None
    window = getattr(host, "_window", None) if host is not None else None
    manager = getattr(window, "user_manager", None) if window is not None else None
    if manager is None:
        manager = getattr(request.app.state, "user_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="User manager not available")
    return manager


@router.post("/login")
def login(payload: LoginRequest, request: Request) -> dict[str, Any]:
    from app.auth.providers import LocalAuthProvider

    manager = _user_manager(request)
    try:
        result = LocalAuthProvider().authenticate(
            payload.username.strip(),
            payload.password,
            manager,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid username or password.") from exc
    ok = bool(result[0]) if isinstance(result, (tuple, list)) else False
    role = result[1] if isinstance(result, (tuple, list)) and len(result) > 1 else None
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    username = payload.username.strip()
    token = issue_access_token(
        username=username,
        role=str(role or "viewer"),
        secret=str(request.app.state.jwt_secret),
    )
    return {"token": token, "username": username, "role": str(role or "viewer")}


@router.post("/logout")
def logout(
    request: Request,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, str]:
    blacklist = getattr(request.app.state, "token_blacklist", None)
    if blacklist is None:
        request.app.state.token_blacklist = set()
        blacklist = request.app.state.token_blacklist
    token = str(identity.get("token") or "")
    jti = identity.get("jti")
    if token:
        blacklist.add(token)
    if jti:
        blacklist.add(str(jti))
    return {"status": "logged_out"}


@router.get("/me")
def me(identity: dict[str, Any] = Depends(require_v1_auth)) -> dict[str, Optional[str]]:
    return {
        "username": identity.get("username"),
        "role": identity.get("role"),
        "kind": identity.get("kind"),
    }
