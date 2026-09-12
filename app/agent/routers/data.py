"""Load local telemetry files and preview the current tab dataset."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agent.routers import attached_host, operator_scope, raise_from_host, require_v1_auth

router = APIRouter(prefix="/tabs", tags=["data"])


class LoadDataBody(BaseModel):
    file_path: str = Field(min_length=1, max_length=2000)
    file_type: Literal["csv", "json"] = "csv"


@router.post("/{tab_id}/data/load", status_code=202)
def load_data(
    tab_id: str,
    payload: LoadDataBody,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        return raise_from_host(
            host.load_tab_data(
                tab_id,
                file_path=payload.file_path,
                file_type=payload.file_type,
                requested_by=requested_by,
            )
        )


@router.get("/{tab_id}/data/preview")
def preview_data(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        result = host.get_tab_data_preview(tab_id, limit=20, head=True)
    if not result.get("ok"):
        return raise_from_host(result)
    return {
        "tab_id": tab_id,
        "columns": result.get("columns") or [],
        "row_count": result.get("row_count") or 0,
        "rows": result.get("rows") or [],
        "status": result.get("status") or "ok",
    }
