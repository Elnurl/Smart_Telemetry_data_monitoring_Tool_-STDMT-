# ADR-003 — Headless ToolHost HTTP API (JWT + routers)

**Status:** Accepted (2026-08)  
**Context:** Sprint 3 — headless ToolHost for local scripts/tests and optional
future clients; **PyQt remains the operator UI**.

## Decision

Keep the existing FastAPI process as the **headless ToolHost**. Add versioned
routers under `/v1/auth`, `/v1/tabs`, `/v1/tabs/{id}/models`, and
`/v1/tabs/{id}/data`. Authenticate those routes with a **local HS256 JWT**
issued by `LocalAuthProvider`. Leave legacy `/tabs`, `/health`, `/agent/*`,
and `/v1/ops/*` unchanged for backward compatibility.

## Why this shape

1. **Stable local contract** — curl/tests/automation can call REST without
   driving the GUI; a later gRPC layer can map the same ToolHost methods.
2. **Loopback only** — bind remains `127.0.0.1`. No external exposure, no
   WebSocket/streaming in this sprint.
3. **Existing RBAC** — JWT `sub` is applied as `current_username` for
   `create_tab` / `delete_tab` / train / data checks. Tab delete allows
   **admin or tab owner** (`created_by`).
4. **Qt thread boundary** — mutations still queue through
   `_QtOperationDispatcher`. The API never touches widgets from the uvicorn
   thread.

## Auth contract

| Item | Choice |
|------|--------|
| Login | `POST /v1/auth/login` → `LocalAuthProvider.authenticate()` |
| Token | PyJWT HS256, 8h TTL, secret `STDMS_JWT_SECRET` (else random per process) |
| Logout | In-memory token/`jti` blacklist (cleared on restart) |
| Header | `Authorization: Bearer <token>` |
| Legacy static token | Still valid on `/tabs` and `/v1/ops/*`; also accepted on new `/v1/tabs` |

`/v1/auth/login` is public. `/health` and `/v1/health` stay public liveness checks.
Prefer **`POST /v1/auth/login`** then **`/v1/tabs/*`** for new clients; legacy `/tabs`
and `/agent/*` remain authenticated but are soft-deprecated (`preferred_api: "/v1"`
in `/health`).

## Consequences

- New operator clients should prefer JWT login over `STDMS_AGENT_API_TOKEN`.
- Training returns `{"status": "training_started", "job_id"}` immediately;
  poll `/operations/{job_id}` (legacy) for completion.
- Data load rejects path traversal: resolved files must stay under the process
  cwd, `data/`, or the tab `data_folder`.
- A future gRPC seam can map these HTTP paths without changing them.

## Not in this ADR

gRPC, WebSocket streaming, and any second desktop UI (PyQt stays canonical).
