"""Top-level tab widgets package."""

from app.tabs.custom_tab.workers import TabMonitoringWorker, TabTrainWorker

__all__ = [
    "CustomMonitoringTab",
    "TabMonitoringWorker",
    "TabTrainWorker",
    "configure_custom_tab",
]


def __getattr__(name: str):
    if name in ("CustomMonitoringTab", "configure_custom_tab"):
        from app.tabs.custom_tab.widget import CustomMonitoringTab, configure_custom_tab

        return CustomMonitoringTab if name == "CustomMonitoringTab" else configure_custom_tab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
