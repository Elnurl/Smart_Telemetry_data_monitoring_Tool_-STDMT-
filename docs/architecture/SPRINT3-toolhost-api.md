# Sprint 3 — Authenticated ToolHost Operator API

## Goal

Turn the local FastAPI process into a **headless ToolHost** for scripts, tests,
and optional future clients. **PyQt remains the operator UI.** Bind remains
loopback only.

## Security model

- Bind: `127.0.0.1` / loopback only.
- New `/v1/auth` and `/v1/tabs*` routes: JWT (`STDMS_JWT_SECRET`, 8h HS256).
- Login uses `LocalAuthProvider.authenticate()`. Logout blacklists the token
  in memory (reset on process restart).
- Legacy `/tabs`, `/agent/*`, `/v1/ops/*`: static `STDMS_AGENT_API_TOKEN`
  (minimum 24 characters), unchanged.
- Comparison: constant-time `hmac.compare_digest` for the static token.
- Authorization: mutating/data operations apply the JWT username to STDMS RBAC.
  Tab delete is admin **or** tab owner (`created_by`).
- Audit: queued/completed/failed mutations go through the existing service audit writer.
- Agent `propose_*` remains approval-gated. Direct `/v1/tabs` mutations are a
  separate authenticated operator-control API.
- Config responses redact keys containing `password`, `secret`, `token`, or `api_key`.

## Contract

### JWT operator API (preferred — `/v1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/auth/login` | Username/password → `{token, username, role}` |
| POST | `/v1/auth/logout` | Blacklist current token |
| GET | `/v1/auth/me` | Identity from token |
| GET | `/v1/tabs` | Snapshot list (`list_tabs` / `get_all_snapshots`) |
| POST | `/v1/tabs` | Create tab (requires existing `data_folder`) |
| DELETE | `/v1/tabs/{tab_id}` | Delete tab (admin or owner) |
| PUT | `/v1/tabs/{tab_id}/config` | Update tab config (no GUI dialog) |
| POST | `/v1/tabs/{tab_id}/start` | Start monitoring |
| POST | `/v1/tabs/{tab_id}/stop` | Stop monitoring |
| GET | `/v1/tabs/{tab_id}/metrics` | Async pipeline metrics for that tab |
| GET | `/v1/tabs/{tab_id}/models` | Model list with `trained` + metrics |
| POST | `/v1/tabs/{tab_id}/models/train` | Background train → `{status, job_id}` |
| GET | `/v1/tabs/{tab_id}/models/metrics` | Tab models + registry recents |
| POST | `/v1/tabs/{tab_id}/data/load` | Load local CSV/JSON (path-traversal checked) |
| GET | `/v1/tabs/{tab_id}/data/preview` | First 20 rows + columns + `row_count` |

### Legacy (unchanged)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/v1/health` | Public liveness/security metadata |
| GET/POST | `/tabs`, `/v1/ops/tabs` | List/create tabs (static bearer) |
| GET/PATCH/DELETE | `/tabs/{tab_id}` | Read/update/delete a tab |
| POST | `/tabs/{tab_id}/train` | Queue model training |
| GET | `/tabs/{tab_id}/data/preview` | Loaded data preview (tail, max 200) |
| POST | `/agent/run`, `/v1/agent/run` | Propose-only agent path |

Mutations return HTTP `202` and an `operation_id` (train also returns `job_id`).
Status transitions are `queued → running → completed|failed`. The in-memory
operation registry is bounded to 500 entries.

## Qt thread boundary

FastAPI runs in a worker thread. `MainWindowToolHost` owns a Qt signal dispatcher
created on the GUI thread. Create/update/delete/train/start/stop/load callbacks
are emitted through that dispatcher; API threads never mutate widgets directly.

## Example

```powershell
$env:STDMS_JWT_SECRET = "replace-with-a-long-local-secret"
.\run.bat

$login = Invoke-RestMethod http://127.0.0.1:8765/v1/auth/login `
  -Method Post -ContentType "application/json" `
  -Body '{"username":"admin","password":"..."}'
$headers = @{ Authorization = "Bearer $($login.token)" }
Invoke-RestMethod http://127.0.0.1:8765/v1/tabs -Headers $headers
```

Legacy static token still works:

```powershell
$env:STDMS_AGENT_API_TOKEN = "replace-with-a-random-token-at-least-24-chars"
$headers = @{ Authorization = "Bearer $env:STDMS_AGENT_API_TOKEN" }
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/tabs -Headers $headers
Invoke-RestMethod http://127.0.0.1:8765/agent/run -Method Post `
  -Headers $headers -ContentType "application/json" `
  -Body '{"tab_id":null,"task":"summarize"}'
```

## Verification

```powershell
py -3.10 -m pytest tests/unit/test_agent_bridge.py tests/unit/test_toolhost_v1.py -q
```

## Sprint 4 seam

The HTTP models and ToolHost methods are transport-neutral boundaries. Sprint 4
can map equivalent protobuf RPCs onto the same ToolHost without moving ML or Qt
code into the Tauri/Avalonia client. WebSocket/streaming is also deferred.

See [ADR-003](ADR-003-toolhost-http-api.md).
