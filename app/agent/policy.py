"""Air-gap / data-security policy for the Instrumentation Agent.

Hard rules for classified / offline environments:
- Telemetry and tab data never leave the machine.
- Agent API binds only to loopback (127.0.0.1 / ::1).
- No cloud LLM, no remote APIs, no third-party agent processes.
- Optional local LLM is OFF by default and must stay on loopback if ever enabled.
"""

from __future__ import annotations

import ipaddress
from typing import FrozenSet

# Only these bind addresses are allowed for the agent HTTP surface.
LOOPBACK_HOSTS: FrozenSet[str] = frozenset({"127.0.0.1", "localhost", "::1"})

# Default: fully offline, no LLM process, no outbound calls.
DEFAULT_AIR_GAP = True
DEFAULT_ALLOW_LLM = False


def is_loopback_host(host: str) -> bool:
    value = (host or "").strip().lower()
    if value in LOOPBACK_HOSTS:
        return True
    # Strip brackets from IPv6 literals: [::1]
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def assert_loopback_bind(host: str) -> str:
    """Return a safe bind host or raise if non-loopback was requested."""
    candidate = (host or "127.0.0.1").strip() or "127.0.0.1"
    if not is_loopback_host(candidate):
        raise ValueError(
            f"Agent API must bind to loopback only (got {candidate!r}). "
            "Refusing non-local bind in air-gap mode."
        )
    # Normalize localhost → 127.0.0.1 for uvicorn consistency
    if candidate.lower() == "localhost":
        return "127.0.0.1"
    return candidate


def is_loopback_url(url: str) -> bool:
    """True only for http(s)://127.0.0.1|localhost|::1/..."""
    from urllib.parse import urlparse

    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    return is_loopback_host(host)
