# Incremental Monolith Refactor

This package is the new modular home for the legacy single-file app.

## Target module boundaries

- `ui/` - Qt views/widgets and interaction logic
- `security/` - cryptography, session lock, security utilities
- `ingestion/` - file/API/DB ingestion adapters
- `models/` - anomaly and forecasting model implementations
- `alerts/` - alert routing, escalation, policy engine
- `auth/` - local/OIDC/SAML auth providers
- `storage/` - SQLite repositories and persistence helpers
- `config/` - typed settings and environment/file overrides
- `services/` - service-layer orchestration between UI and domain logic

## Backward compatibility strategy

1. Keep `tool_PyQt copy 2_1 copy 8_copy_main.py` as the executable entrypoint.
2. Extract components into `app/*` and import them with safe fallbacks.
3. Route new logic through service-layer interfaces.
4. Remove legacy inline code only after parity tests pass.

## Current extracted pieces

- Typed settings loader: `app/config/settings.py`
- Service layer scaffold: `app/services/service_layer.py`
- Alerts policy module: `app/alerts/policy.py`
- Auth providers module: `app/auth/providers.py`
- Audit logging module: `app/storage/audit.py`
- Package structure for all target modules

