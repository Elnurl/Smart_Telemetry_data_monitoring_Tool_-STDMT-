"""Local FastAPI surface for the Instrumentation Agent (localhost only)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agent.audit import AgentAuditLog
from app.agent.bridge import get_tool_host
from app.agent.policy import assert_loopback_bind
from app.agent.runner import AgentRunner
from app.agent.tools import list_tools

logger = logging.getLogger("STDMS.Agent.Server")


class AgentRunRequest(BaseModel):
    tab_id: Optional[str] = None
    task: str = Field(default="summarize", max_length=500)


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


def create_app(audit: AgentAuditLog, runner: AgentRunner):
    try:
        from fastapi import FastAPI, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError(
            "fastapi is required for the agent bridge. Install with: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(
        title="STDMS Instrumentation Agent",
        description="Phase 4 instrumentation agent — tool chain, propose drafts, local RAG",
        version="0.4.0",
    )

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
            "tool_host_attached": host is not None,
            "mode": "tool_chain_propose_rag",
            "air_gap": True,
            "llm_enabled": bool(getattr(runner, "allow_llm", False)),
            "data_egress": "none",
            "bind": "loopback_only",
            "mutating_tools": True,
            "mutating_note": "propose_* only — human approval required",
            "knowledge": ks,
        }

    @app.get("/tools")
    def tools():
        return {"tools": list_tools(include_mutating=True)}

    @app.get("/tabs")
    def tabs():
        host = get_tool_host()
        if host is None:
            raise HTTPException(status_code=503, detail="Tool host not attached")
        return {"tabs": host.list_tabs()}

    @app.get("/tabs/{tab_id}/snapshot")
    def tab_snapshot(tab_id: str):
        host = get_tool_host()
        if host is None:
            raise HTTPException(status_code=503, detail="Tool host not attached")
        snap = host.get_snapshot(tab_id)
        if snap is None:
            raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
        return {"tab_id": tab_id, "snapshot": snap}

    @app.get("/tabs/{tab_id}/history")
    def tab_history(tab_id: str, n: int = Query(default=50, ge=1, le=500)):
        host = get_tool_host()
        if host is None:
            raise HTTPException(status_code=503, detail="Tool host not attached")
        if host.get_snapshot(tab_id) is None:
            raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
        return {"tab_id": tab_id, "history": host.get_history(tab_id, n=n)}

    @app.get("/retrain-signals")
    def retrain_signals():
        host = get_tool_host()
        if host is None:
            raise HTTPException(status_code=503, detail="Tool host not attached")
        return {"signals": host.get_pending_retrain_signals()}

    @app.post("/agent/run")
    def agent_run(payload: AgentRunRequest):
        return runner.run(tab_id=payload.tab_id, task=payload.task)

    @app.get("/agent/decisions")
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
) -> AgentServerHandle:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for the agent bridge. Install with: pip install fastapi uvicorn"
        ) from exc

    host = assert_loopback_bind(host)
    app = create_app(audit=audit, runner=runner)
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
