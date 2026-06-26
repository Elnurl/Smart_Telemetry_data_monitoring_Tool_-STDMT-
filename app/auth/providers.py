from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, Tuple
from urllib import request as urllib_request


class BaseAuthProvider:
    provider_name = "base"

    def authenticate(self, username: str, password: str, user_manager: Any):
        raise NotImplementedError


class LocalAuthProvider(BaseAuthProvider):
    provider_name = "local"

    def authenticate(self, username: str, password: str, user_manager: Any):
        return user_manager._authenticate_local(username, password)


class OIDCAuthProvider(BaseAuthProvider):
    provider_name = "oidc"

    def _map_role(self, raw_roles, mapping: Dict[str, str], default_role: str = "viewer") -> str:
        if raw_roles is None:
            return default_role
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]
        for role in raw_roles:
            mapped = mapping.get(str(role).lower())
            if mapped:
                return mapped
        return default_role

    def authenticate(self, username: str, password: str, user_manager: Any) -> Tuple[bool, Any, str, bool, Dict[str, Any]]:
        cfg = (user_manager.auth_config or {}).get("oidc", {})
        if not cfg.get("enabled"):
            return False, None, "OIDC provider is disabled.", False, {"auth_provider": "oidc"}

        token_endpoint = str(cfg.get("token_endpoint", "")).strip()
        userinfo_endpoint = str(cfg.get("userinfo_endpoint", "")).strip()
        client_id = str(cfg.get("client_id", "")).strip()
        client_secret = str(cfg.get("client_secret", "")).strip()
        if not token_endpoint or not userinfo_endpoint or not client_id:
            return False, None, "OIDC is not fully configured.", False, {"auth_provider": "oidc"}

        post_data = {
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": client_id,
            "scope": str(cfg.get("scope", "openid profile email")),
        }
        if client_secret:
            post_data["client_secret"] = client_secret

        encoded = urllib.parse.urlencode(post_data).encode("utf-8")
        req = urllib_request.Request(
            token_endpoint,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=15) as resp:
                token_payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False, None, "OIDC authentication failed.", False, {"auth_provider": "oidc"}

        access_token = token_payload.get("access_token")
        if not access_token:
            return False, None, "OIDC access token missing in response.", False, {"auth_provider": "oidc"}

        userinfo_req = urllib_request.Request(
            userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        try:
            with urllib_request.urlopen(userinfo_req, timeout=15) as resp:
                userinfo = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False, None, "OIDC user info retrieval failed.", False, {"auth_provider": "oidc"}

        username_claim = str(cfg.get("username_claim", "preferred_username"))
        email_claim = str(cfg.get("email_claim", "email"))
        role_claim = str(cfg.get("role_claim", "roles"))
        role_mapping = cfg.get("role_mapping", {}) or {}

        resolved_username = str(userinfo.get(username_claim) or username).strip()
        resolved_email = str(userinfo.get(email_claim) or "").strip()
        raw_roles = userinfo.get(role_claim)
        resolved_role = self._map_role(raw_roles, role_mapping, cfg.get("default_role", "viewer"))

        user_manager.ensure_identity_record(
            username=resolved_username,
            role=resolved_role,
            email=resolved_email,
            auth_provider="oidc",
        )

        return True, resolved_role, "OIDC login successful.", False, {
            "auth_provider": "oidc",
            "resolved_username": resolved_username,
            "email": resolved_email,
            "oidc_subject": userinfo.get("sub"),
            "raw_roles": raw_roles,
        }

