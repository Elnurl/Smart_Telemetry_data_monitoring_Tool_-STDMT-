from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
from typing import Any, Callable, Dict, Optional


def load_alert_routing_config(config_file: str, logger: Any):
    default_config = {
        "cooldown_seconds": 60,
        "dedup_window_seconds": 300,
        "escalation_unacked_minutes": 15,
        "routes": {
            "info": ["email"],
            "warning": ["email", "webhook"],
            "critical": ["email", "webhook", "pagerduty"],
        },
        "channels": {
            "email": {"enabled": True},
            "slack": {"enabled": False, "webhook_url": ""},
            "webhook": {"enabled": False, "url": ""},
            "pagerduty": {
                "enabled": False,
                "events_api_url": "https://events.pagerduty.com/v2/enqueue",
                "routing_key": "",
            },
        },
    }

    try:
        import os

        if not os.path.exists(config_file):
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            return default_config

        with open(config_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = default_config.copy()
            merged.update(loaded)
            merged["routes"] = {**default_config["routes"], **loaded.get("routes", {})}
            merged["channels"] = {**default_config["channels"], **loaded.get("channels", {})}
            return merged
    except Exception as e:
        logger.error(f"Failed to load alert routing config: {e}")

    return default_config


class AlertPolicyManager:
    def __init__(
        self,
        db_path: str,
        config_loader: Callable[[], Dict[str, Any]],
        dispatchers: Optional[Dict[str, Callable[..., None]]] = None,
    ):
        self.db_path = db_path
        self.config_loader = config_loader
        self.dispatchers = dispatchers or {}
        self.config = self.config_loader()

    def reload_config(self):
        self.config = self.config_loader()
        return self.config

    def _db(self):
        return sqlite3.connect(self.db_path)

    def _fingerprint(self, severity, message, payload):
        model_name = str((payload or {}).get("model_name", "unknown"))
        raw = f"{str(severity).lower()}|{model_name}|{str(message).strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _log_event(self, conn, incident_id, event_type, severity, channel=None, status="ok", details=None):
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO alert_lifecycle_events
            (incident_id, event_time, event_type, severity, channel, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(incident_id),
                datetime.datetime.utcnow().isoformat(),
                str(event_type),
                str(severity),
                channel,
                str(status),
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )

    def _should_route_with_cooldown(self, conn, incident_id, cooldown_seconds):
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT event_time FROM alert_lifecycle_events
            WHERE incident_id = ? AND event_type IN ('triggered', 'escalated')
            ORDER BY id DESC LIMIT 1
            """,
            (int(incident_id),),
        )
        row = cursor.fetchone()
        if not row:
            return True
        try:
            last_ts = datetime.datetime.fromisoformat(row[0])
            return (datetime.datetime.utcnow() - last_ts).total_seconds() >= float(cooldown_seconds)
        except Exception:
            return True

    def process_alert(self, severity, message, payload=None):
        self.reload_config()
        payload = (payload or {}).copy()
        severity = str(severity).lower()
        fingerprint = self._fingerprint(severity, message, payload)
        dedup_window = int(self.config.get("dedup_window_seconds", 300))
        cooldown_seconds = int(self.config.get("cooldown_seconds", 60))
        now_iso = datetime.datetime.utcnow().isoformat()
        next_escalation_at = (
            datetime.datetime.utcnow()
            + datetime.timedelta(minutes=int(self.config.get("escalation_unacked_minutes", 15)))
        ).isoformat()

        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, last_seen
                FROM alert_incidents
                WHERE fingerprint = ? AND status IN ('open', 'escalated')
                ORDER BY id DESC LIMIT 1
                """,
                (fingerprint,),
            )
            row = cursor.fetchone()
            incident_id = None
            action = "created"

            if row:
                incident_id = int(row[0])
                last_seen = row[1]
                within_dedup = False
                try:
                    within_dedup = (
                        datetime.datetime.utcnow() - datetime.datetime.fromisoformat(last_seen)
                    ).total_seconds() <= dedup_window
                except Exception:
                    within_dedup = True

                if within_dedup:
                    action = "deduplicated"
                    cursor.execute(
                        """
                        UPDATE alert_incidents
                        SET last_seen = ?, occurrence_count = occurrence_count + 1, last_payload = ?
                        WHERE id = ?
                        """,
                        (now_iso, json.dumps(payload, ensure_ascii=False), incident_id),
                    )
                    self._log_event(conn, incident_id, "deduplicated", severity, status="ok", details={"message": message})
                else:
                    cursor.execute(
                        """
                        UPDATE alert_incidents
                        SET status = 'open', severity = ?, message = ?, last_seen = ?, occurrence_count = occurrence_count + 1,
                            next_escalation_at = ?, last_payload = ?
                        WHERE id = ?
                        """,
                        (severity, str(message), now_iso, next_escalation_at, json.dumps(payload, ensure_ascii=False), incident_id),
                    )
                    self._log_event(conn, incident_id, "reopened", severity, status="ok", details={"message": message})
            else:
                cursor.execute(
                    """
                    INSERT INTO alert_incidents
                    (fingerprint, status, severity, message, model_name, first_seen, last_seen,
                     occurrence_count, acknowledged_by, acknowledged_at, escalated, next_escalation_at, last_payload)
                    VALUES (?, 'open', ?, ?, ?, ?, ?, 1, NULL, NULL, 0, ?, ?)
                    """,
                    (
                        fingerprint,
                        severity,
                        str(message),
                        str(payload.get("model_name", "Unknown")),
                        now_iso,
                        now_iso,
                        next_escalation_at,
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
                incident_id = int(cursor.lastrowid)
                self._log_event(conn, incident_id, "created", severity, status="ok", details={"message": message})

            should_route = self._should_route_with_cooldown(conn, incident_id, cooldown_seconds)
            if not should_route:
                self._log_event(
                    conn,
                    incident_id,
                    "suppressed_cooldown",
                    severity,
                    status="suppressed",
                    details={"cooldown_seconds": cooldown_seconds},
                )
                conn.commit()
                return {"incident_id": incident_id, "action": action, "routed": False}

            routes = list(self.config.get("routes", {}).get(severity, []))
            for route in routes:
                route_status = self._dispatch_route(route, incident_id, severity, message, payload, escalated=False)
                self._log_event(conn, incident_id, "triggered", severity, channel=route, status=route_status["status"], details=route_status)

            conn.commit()
            return {"incident_id": incident_id, "action": action, "routed": True, "routes": routes}

    def acknowledge(self, incident_id, acknowledged_by):
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE alert_incidents
                SET status = 'acknowledged', acknowledged_by = ?, acknowledged_at = ?
                WHERE id = ? AND status IN ('open', 'escalated')
                """,
                (str(acknowledged_by), datetime.datetime.utcnow().isoformat(), int(incident_id)),
            )
            changed = cursor.rowcount > 0
            if changed:
                self._log_event(
                    conn,
                    incident_id,
                    "acknowledged",
                    "info",
                    status="ok",
                    details={"acknowledged_by": acknowledged_by},
                )
            conn.commit()
            return changed

    def run_escalation_cycle(self):
        self.reload_config()
        now = datetime.datetime.utcnow()
        escalated_ids = []
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, severity, message, model_name, next_escalation_at, last_payload
                FROM alert_incidents
                WHERE status IN ('open', 'escalated')
                  AND acknowledged_at IS NULL
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                incident_id = int(row[0])
                severity = str(row[1]).lower()
                message = str(row[2])
                next_escalation_at = row[4]
                try:
                    if not next_escalation_at or datetime.datetime.fromisoformat(next_escalation_at) > now:
                        continue
                except Exception:
                    pass

                payload = {}
                try:
                    payload = json.loads(row[5]) if row[5] else {}
                except Exception:
                    payload = {"model_name": row[3]}

                routes = list(self.config.get("routes", {}).get(severity, []))
                for route in routes:
                    route_status = self._dispatch_route(
                        route, incident_id, severity, f"[ESCALATED] {message}", payload, escalated=True
                    )
                    self._log_event(conn, incident_id, "escalated", severity, channel=route, status=route_status["status"], details=route_status)

                next_time = (
                    now + datetime.timedelta(minutes=int(self.config.get("escalation_unacked_minutes", 15)))
                ).isoformat()
                cursor.execute(
                    """
                    UPDATE alert_incidents
                    SET status = 'escalated', escalated = 1, last_seen = ?, next_escalation_at = ?
                    WHERE id = ?
                    """,
                    (now.isoformat(), next_time, incident_id),
                )
                escalated_ids.append(incident_id)

            conn.commit()
        return escalated_ids

    def _dispatch_route(self, route, incident_id, severity, message, payload, escalated=False):
        route = str(route).lower()
        channel_cfg = (self.config.get("channels", {}).get(route, {}) or {})
        if not channel_cfg.get("enabled", False):
            return {"status": "skipped", "reason": "disabled"}

        dispatcher = self.dispatchers.get(route)
        if not dispatcher:
            return {"status": "error", "reason": "missing_dispatcher"}

        try:
            dispatcher(
                incident_id=incident_id,
                severity=severity,
                message=message,
                payload=payload,
                channel_config=channel_cfg,
                escalated=escalated,
            )
            return {"status": "ok", "escalated": bool(escalated)}
        except Exception as e:
            return {"status": "error", "error": str(e), "escalated": bool(escalated)}

