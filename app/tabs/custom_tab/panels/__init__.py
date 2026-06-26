"""Custom tab UI panel builders (Slice B6 — complete).

Model training (canonical)
--------------------------
**``tab.models``** via Model Development and Analysis Train redirect
(``train_from_analysis_panel`` → ``CustomMonitoringTab.train_model``).

Analysis panel Save/Load still use legacy handlers on the main window; on custom tabs
those map to tab-native ``save_model`` / ``load_model`` via ``_TAB_NATIVE_METHODS`` bind rules
for direct calls, but Analysis buttons still use ``_analysis_slot`` (legacy invoke) except Train.

Legacy panels
-------------
``data_import.py``, ``analysis_ml.py``, ``visualization.py`` — wired via ``LegacyPanelSlots``
from ``configure_custom_tab(build_legacy_panel_slots=...)``.
"""

from app.tabs.custom_tab.panels.header_config import build_header_config_group
from app.tabs.custom_tab.panels.models_page import build_models_page
from app.tabs.custom_tab.panels.quick_actions import build_quick_actions_page
from app.tabs.custom_tab.panels.shell import build_nav_shell, connect_nav_pages

__all__ = [
    "build_header_config_group",
    "build_models_page",
    "build_quick_actions_page",
    "build_nav_shell",
    "connect_nav_pages",
]
