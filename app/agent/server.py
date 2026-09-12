"""Local FastAPI surface for the Instrumentation Agent (localhost only)."""

from __future__ import annotations

import logging
import hmac
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agent.audit import AgentAuditLog
from app.agent.bridge import get_tool_host
from app.agent.jwt_tokens import resolve_jwt_secret
from app.agent.policy import assert_loopback_bind
from app.agent.runner import AgentRunner
from app.agent.tools import list_tools

logger = logging.getLogger("STDMS.Agent.Server")


class AgentRunRequest(BaseModel):
    tab_id: Optional[str] = None
    task: str = Field(default="summarize", max_length=500)


class TabCreateRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="api", max_length=100)


class TabUpdateRequest(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="api", max_length=100)


class TrainRequest(BaseModel):
    model_id: Optional[str] = Field(default=None, max_length=100)
    requested_by: str = Field(default="api", max_length=100)


def resolve_api_token(explicit: Optional[str] = None) -> str:
    """Resolve a per-process API bearer token without logging its value."""
    token = (explicit or os.getenv("STDMS_AGENT_API_TOKEN") or "").strip()
    if token:
        if len(token) < 24:
            raise ValueError("STDMS agent API token must contain at least 24 characters")
        return token
    return secrets.token_urlsafe(32)


@dataclass
class AgentServerHandle:
    host: str
    port: int
    thread: threading.Thread
    server: Any

    def stop(self) -> None:
        try:
            self.server.should_exit = True
        except Exception:
            pass


def create_app(
    audit: AgentAuditLog,
    runner: AgentRunner,
    *,
    api_token: Optional[str] = None,
    jwt_secret: Optional[str] = None,
):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError(
            "fastapi is required for the agent bridge. Install with: pip install fastapi uvicorn"
        ) from exc

    token = resolve_api_token(api_token)
    app = FastAPI(
        title="STDMS SatOps Agent",
        description="Local operator API — tools, tabs, training, data, agent and RAG",
        version="0.6.0",
    )
    app.state.api_token = token
    app.state.jwt_secret = resolve_jwt_secret(jwt_secret)
    app.state.token_blacklist = set()

    from app.agent.routers import auth as auth_router
    from app.agent.routers import data as data_router
    from app.agent.routers import models as models_router
    from app.agent.routers import tabs as tabs_router
    from fastapi.middleware.cors import CORSMiddleware

    app.include_router(auth_router.router, prefix="/v1")
    app.include_router(tabs_router.router, prefix="/v1")
    app.include_router(models_router.router, prefix="/v1")
    app.include_router(data_router.router, prefix="/v1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "tauri://localhost",
            "https://tauri.localhost",
            "http://tauri.localhost",
            "https://localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        raw = supplied.strip()
        valid_static = (
            scheme.lower() == "bearer"
            and bool(raw)
            and hmac.compare_digest(raw, token)
        )
        if valid_static:
            return
        if scheme.lower() == "bearer" and raw:
            blacklist = getattr(app.state, "token_blacklist", set())
            if raw not in blacklist:
                try:
                    from app.agent.jwt_tokens import decode_access_token

                    payload = decode_access_token(raw, str(app.state.jwt_secret))
                    jti = str(payload.get("jti") or "")
                    if not jti or jti not in blacklist:
                        return
                except ValueError:
                    pass
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    protected = [Depends(require_auth)]

    def attached_host():
        host = get_tool_host()
        if host is None:
            raise HTTPException(status_code=503, detail="Tool host not attached")
        return host

    def accepted_or_error(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("ok"):
            return result
        status = str(result.get("status") or "")
        if status in {"tab_not_found", "model_not_found"}:
            code = 404
        elif status == "forbidden":
            code = 403
        elif status == "invalid_config":
            code = 422
        else:
            code = 409
        raise HTTPException(status_code=code, detail=result)

    @app.get("/v1/health")
    @app.get("/health")
    def health():
        host = get_tool_host()
        try:
            from app.agent.rag.retrieve import knowledge_status as rag_status

            ks = rag_status()
        except Exception:
            ks = {"status": "unavailable"}
        return {
            "ok": True,
            "phase": 4,
            "sprint": 3,
            "api_version": "0.6.0",
            "preferred_api": "/v1",
            "legacy_routes": "deprecated_but_supported",
            "tool_host_attached": host is not None,
            "mode": "authenticated_operator_api",
            "air_gap": True,
            "llm_enabled": bool(getattr(runner, "allow_llm", False)),
            "data_egress": "none",
            "bind": "loopback_only",
            "auth": "bearer_required",
            "mutating_tools": True,
            "mutating_note": (
                "Agent propose_* remains approval-gated; authenticated operator endpoints "
                "queue direct GUI operations under current STDMS RBAC. "
                "Prefer /v1/auth/login + /v1/tabs over legacy /tabs."
            ),
            "knowledge": ks,
        }

    @app.get("/v1/agent/tools", dependencies=protected)
    @app.get("/tools", dependencies=protected)
    def tools():
        return {"tools": list_tools(include_mutating=True)}

    @app.get("/v1/ops/tabs", dependencies=protected)
    @app.get("/tabs", dependencies=protected)
    def tabs():
        host = attached_host()
        return {"tabs": host.list_tabs()}

    @app.post("/v1/ops/tabs", status_code=202, dependencies=protected)
    @app.post("/tabs", status_code=202, dependencies=protected)
    def create_tab(payload: TabCreateRequest):
        return accepted_or_error(
            attached_host().create_tab(payload.config, requested_by=payload.requested_by)
        )

    @app.get("/v1/ops/tabs/{tab_id}", dependencies=protected)
    @app.get("/tabs/{tab_id}", dependencies=protected)
    def tab_definition(tab_id: str):
        definition = attached_host().get_tab_definition(tab_id)
        if definition is None:
            raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
        return definition

    @app.patch("/v1/ops/tabs/{tab_id}", status_code=202, dependencies=protected)
    @app.patch("/tabs/{tab_id}", status_code=202, dependencies=protected)
    def update_tab(tab_id: str, payload: TabUpdateRequest):
        return accepted_or_error(
            attached_host().update_tab(
                tab_id,
                payload.updates,
                requested_by=payload.requested_by,
            )
        )

    @app.delete("/v1/ops/tabs/{tab_id}", status_code=202, dependencies=protected)
    @app.delete("/tabs/{tab_id}", status_code=202, dependencies=protected)
    def delete_tab(tab_id: str, requested_by: str = Query(default="api", max_length=100)):
        return accepted_or_error(
            attached_host().delete_tab(tab_id, requested_by=requested_by)
        )

    @app.get("/v1/ops/tabs/{tab_id}/snapshot", dependencies=protected)
    @app.get("/tabs/{tab_id}/snapshot", dependencies=protected)
    def tab_snapshot(tab_id: str):
        host = attached_host()
        snap = host.get_snapshot(tab_id)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
        return {"tab_id": tab_id, "snapshot": snap}

    @app.get("/v1/ops/tabs/{tab_id}/history", dependencies=protected)
    @app.get("/tabs/{tab_id}/history", dependencies=protected)
    def tab_history(tab_id: str, n: int = Query(default=50, ge=1, le=500)):
        host = attached_host()
        if host.get_snapshot(tab_id) is None:
            raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
        return {"tab_id": tab_id, "history": host.get_history(tab_id, n=n)}

    @app.get("/v1/ops/tabs/{tab_id}/models", dependencies=protected)
    @app.get("/tabs/{tab_id}/models", dependencies=protected)
    def tab_models(tab_id: str):
        return accepted_or_error(attached_host().list_tab_models(tab_id))

    @app.get("/v1/ops/tabs/{tab_id}/models/{model_id}", dependencies=protected)
    @app.get("/tabs/{tab_id}/models/{model_id}", dependencies=protected)
    def model_metrics(tab_id: str, model_id: str):
        return accepted_or_error(attached_host().get_model_metrics(tab_id, model_id))

    @app.post("/v1/ops/tabs/{tab_id}/train", status_code=202, dependencies=protected)
    @app.post("/tabs/{tab_id}/train", status_code=202, dependencies=protected)
    def train_tab(tab_id: str, payload: TrainRequest):
        return accepted_or_error(
            attached_host().train_tab(
                tab_id,
                model_id=payload.model_id,
                requested_by=payload.requested_by,
            )
        )

    @app.get("/v1/ops/tabs/{tab_id}/data/schema", dependencies=protected)
    @app.get("/tabs/{tab_id}/data/schema", dependencies=protected)
    def tab_data_schema(tab_id: str):
        return accepted_or_error(attached_host().get_tab_data_schema(tab_id))

    @app.get("/v1/ops/tabs/{tab_id}/data/preview", dependencies=protected)
    @app.get("/tabs/{tab_id}/data/preview", dependencies=protected)
    def tab_data_preview(tab_id: str, limit: int = Query(default=50, ge=1, le=200)):
        return accepted_or_error(attached_host().get_tab_data_preview(tab_id, limit=limit))

    @app.get("/v1/ops/operations/{operation_id}", dependencies=protected)
    @app.get("/operations/{operation_id}", dependencies=protected)
    def operation_status(operation_id: str):
        operation = attached_host().get_operation(operation_id)
        if operation is None:
            raise HTTPException(status_code=404, detail=f"Operation not found: {operation_id}")
        return operation

    @app.get("/v1/agent/retrain-signals", dependencies=protected)
    @app.get("/retrain-signals", dependencies=protected)
    def retrain_signals():
        host = attached_host()
        return {"signals": host.get_pending_retrain_signals()}

    @app.get("/v1/agent/drafts", dependencies=protected)
    def pending_drafts():
        host = attached_host()
        drafts = host.list_pending_drafts() if hasattr(host, "list_pending_drafts") else []
        return {"drafts": drafts, "pending_draft_count": len(drafts or [])}

    @app.post("/v1/agent/run", dependencies=protected)
    @app.post("/agent/run", dependencies=protected)
    def agent_run(payload: AgentRunRequest):
        return runner.run(tab_id=payload.tab_id, task=payload.task)

    @app.get("/v1/agent/decisions", dependencies=protected)
    @app.get("/agent/decisions", dependencies=protected)
    def agent_decisions(
        limit: int = Query(default=50, ge=1, le=500),
        tab_id: Optional[str] = None,
    ):
        return {"decisions": audit.list_decisions(limit=limit, tab_id=tab_id)}

    return app


def start_agent_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    audit: AgentAuditLog,
    runner: AgentRunner,
    api_token: Optional[str] = None,
) -> AgentServerHandle:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for the agent bridge. Install with: pip install fastapi uvicorn"
        ) from exc

    host = assert_loopback_bind(host)
    app = create_app(audit=audit, runner=runner, api_token=api_token)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run,
        name="stdms-agent-api",
        daemon=True,
    )
    thread.start()
    logger.info("Agent API started on http://%s:%s", host, port)
    return AgentServerHandle(host=host, port=port, thread=thread, server=server)
