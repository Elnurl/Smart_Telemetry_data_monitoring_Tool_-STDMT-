"""Legacy monolith panel slot bundle (B6 Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List


@dataclass(frozen=True)
class LegacyPanelSlots:
    """Callbacks required to wire Data Import / Analysis / Visualization panels."""

    tool_window: Any
    ensure_panel_helpers: Callable[[Any], None]
    data_slot: Callable[[Any, str], Callable[..., None]]
    data_import_stop_slot: Callable[[Any], Callable[..., None]]
    analysis_slot: Callable[[Any, str], Callable[..., None]]
    viz_slot: Callable[[Any, str], Callable[..., None]]
    invoke_analysis_method: Callable[..., Any]
    invoke_visualization_method: Callable[..., Any]
    get_supported_model_names: Callable[[], List[str]]
    mpl_canvas_cls: Any


def assert_panel_slots_complete(slots: LegacyPanelSlots | None) -> LegacyPanelSlots:
    if slots is None:
        raise RuntimeError("build_legacy_panel_slots is not configured — call configure_custom_tab() first")
    for name in LegacyPanelSlots.__dataclass_fields__:
        if getattr(slots, name) is None:
            raise RuntimeError(f"legacy panel slots incomplete — missing: {name}")
    return slots
