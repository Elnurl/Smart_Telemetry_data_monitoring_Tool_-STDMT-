from __future__ import annotations

import datetime

SCHEDULE_CONTINUOUS = "Continuous"
SCHEDULE_ON_DEMAND = "On-Demand"
SCHEDULE_SCHEDULED = "Scheduled"


def interval_ms_from_parts(hours: int, minutes: int, seconds: int) -> int:
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1_000


def should_run_scheduled_tab(config: dict, last_run_key: str | None) -> bool:
    """Return True when a scheduled tab should run for the current UTC minute."""
    schedule_type = config.get("schedule_type", SCHEDULE_CONTINUOUS)
    if schedule_type != SCHEDULE_SCHEDULED:
        return False

    hour = int(config.get("schedule_utc_hour", 0))
    minute = int(config.get("schedule_utc_minute", 0))
    now = datetime.datetime.utcnow()
    current_key = f"{now.date().isoformat()}T{hour:02d}:{minute:02d}"
    if now.hour != hour or now.minute != minute:
        return False
    return last_run_key != current_key
