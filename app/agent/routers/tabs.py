"""Tab inventory, create/delete/config, start/stop, pipeline metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.agent.routers import attached_host, operator_scope, raise_from_host, require_v1_auth

router = APIRouter(prefix="/tabs", tags=["tabs"])


def _flatten_tab_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = dict(payload)
    name = str(config.get("name") or "").strip()
    title = str(config.get("title") or name).strip()
    if title:
        config["title"] = title
    schedule = config.get("schedule")
    if isinstance(schedule, dict):
        config.setdefault(
            "schedule_type",
            schedule.get("type") or schedule.get("schedule_type"),
        )
        if schedule.get("interval_ms") is not None:
            config.setdefault("interval_ms", schedule.get("interval_ms"))
        if schedule.get("utc_hour") is not None:
            config.setdefault("schedule_utc_hour", schedule.get("utc_hour"))
        if schedule.get("utc_minute") is not None:
            config.setdefault("schedule_utc_minute", schedule.get("utc_minute"))
    return config


@router.get("")
@router.get("/")
def list_tabs(identity: dict[str, Any] = Depends(require_v1_auth)) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        return {"tabs": host.list_tabs()}


@router.post("", status_code=202)
@router.post("/", status_code=202)
def create_tab(
    payload: dict[str, Any],
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="TabConfig object required")
    config = _flatten_tab_payload(payload)
    data_folder = str(config.get("data_folder") or "").strip()
    if not data_folder:
        raise HTTPException(status_code=400, detail="data_folder is required")
    if not Path(data_folder).is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"data_folder does not exist: {data_folder}",
        )
    if not str(config.get("title") or "").strip():
        raise HTTPException(status_code=400, detail="name or title is required")
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        return raise_from_host(host.create_tab(config, requested_by=requested_by))


@router.delete("/{tab_id}", status_code=202)
def delete_tab(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        return raise_from_host(host.delete_tab(tab_id, requested_by=requested_by))


@router.put("/{tab_id}/config", status_code=202)
def update_tab_config(
    tab_id: str,
    payload: dict[str, Any],
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="configuration updates are required")
    updates = _flatten_tab_payload(payload)
    updates.pop("models", None)
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        return raise_from_host(
            host.update_tab(tab_id, updates, requested_by=requested_by)
        )


@router.post("/{tab_id}/start", status_code=202)
def start_tab(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        result = raise_from_host(host.start_tab(tab_id, requested_by=requested_by))
    if result.get("status") == "already_monitoring":
        return result
    return result


@router.post("/{tab_id}/stop", status_code=202)
def stop_tab(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    requested_by = str(identity.get("username") or "api")
    with operator_scope(host, identity):
        return raise_from_host(host.stop_tab(tab_id, requested_by=requested_by))


@router.get("/{tab_id}/metrics")
def tab_metrics(
    tab_id: str,
    limit: int = 50,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        return raise_from_host(host.get_tab_pipeline_metrics(tab_id, limit=limit))


@router.get("/{tab_id}/snapshot")
def tab_snapshot(
    tab_id: str,
    identity: dict[str, Any] = Depends(require_v1_auth),
) -> dict[str, Any]:
    host = attached_host()
    with operator_scope(host, identity):
        snap = host.get_snapshot(tab_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"Tab not found: {tab_id}")
    return {"tab_id": tab_id, "snapshot": snap}
