"""Tab model inventory, background train, and registry metrics."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.agent.routers import attached_host, operator_scope, raise_from_host, require_v1_auth

router = APIRouter(prefix="/tabs", tags=["models"])


class TrainBody(BaseModel):
    model_id: Optional[str] = Field(default=None, max_length=100)


@router.get("/{tab_id}/models")
def list_models(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        return raise_from_host(host.list_tab_models(tab_id))


@router.post("/{tab_id}/models/train", status_code=202)
def train_models(
    tab_id: str,
    payload: Optional[TrainBody] = None,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    model_id = payload.model_id if payload is not None else None
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        result = raise_from_host(
            host.train_tab(tab_id, model_id=model_id, requested_by=requested_by)
        )
    job_id = str(result.get("job_id") or result.get("operation_id") or "")
    return {
        "status": "training_started",
        "job_id": job_id,
        "operation_id": result.get("operation_id"),
        "tab_id": tab_id,
        "model_id": model_id,
    }


@router.get("/{tab_id}/models/metrics")
def model_metrics(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        return raise_from_host(host.get_registry_models(tab_id))
