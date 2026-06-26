from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabVisibilityPolicy:
    can_import_data: bool
    can_process_data: bool
    can_manage_users: bool


def build_tab_visibility_policy(service_layer, username: str) -> TabVisibilityPolicy:
    return TabVisibilityPolicy(
        can_import_data=bool(service_layer.authorize(username, "import_data", resource="data_source:*")),
        can_process_data=bool(service_layer.authorize(username, "process_data", resource="analysis")),
        can_manage_users=bool(service_layer.authorize(username, "manage_users", resource="admin/users")),
    )

