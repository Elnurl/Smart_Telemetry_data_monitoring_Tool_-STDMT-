from __future__ import annotations

import datetime
import getpass
import json
import os
import socket
from typing import Any, Dict, Optional


def get_runtime_source_context() -> Dict[str, Any]:
    hostname = socket.gethostname()
    local_ip = "unknown"
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        pass

    return {
        "hostname": hostname,
        "local_ip": local_ip,
        "os_user": getpass.getuser(),
        "process_id": os.getpid(),
    }


def write_audit_event(
    audit_log_file: str,
    actor: Any,
    action: str,
    resource: str = "*",
    outcome: str = "success",
    details: Optional[Dict[str, Any]] = None,
    auth_provider: str = "local",
    source_context: Optional[Dict[str, Any]] = None,
) -> None:
    event = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "actor": str(actor) if actor is not None else "unknown",
        "action": str(action),
        "resource": str(resource),
        "outcome": str(outcome),
        "auth_provider": str(auth_provider),
        "source": source_context or get_runtime_source_context(),
        "details": details or {},
    }

    os.makedirs(os.path.dirname(audit_log_file), exist_ok=True)
    with open(audit_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

