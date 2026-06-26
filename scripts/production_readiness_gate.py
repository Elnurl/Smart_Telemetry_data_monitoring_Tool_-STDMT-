from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.alerts.policy import AlertPolicyManager
from app.ingestion.data_reader import DataReader
from app.models.health_scoring import compute_health_score


@dataclass
class SLOTargets:
    alert_latency_p95_ms: float = 250.0
    ingestion_success_rate_pct: float = 99.5
    false_positive_rate_pct: float = 2.0


def _init_alert_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT,
            status TEXT,
            severity TEXT,
            message TEXT,
            model_name TEXT,
            first_seen DATETIME,
            last_seen DATETIME,
            occurrence_count INTEGER DEFAULT 1,
            acknowledged_by TEXT,
            acknowledged_at DATETIME,
            escalated INTEGER DEFAULT 0,
            next_escalation_at DATETIME,
            last_payload TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_lifecycle_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER,
            event_time DATETIME,
            event_type TEXT,
            severity TEXT,
            channel TEXT,
            status TEXT,
            details TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _make_config(routes: dict[str, list[str]], channels: dict[str, dict]) -> dict:
    return {
        "cooldown_seconds": 0,
        "dedup_window_seconds": 0,
        "escalation_unacked_minutes": 1,
        "routes": routes,
        "channels": channels,
    }


def run_gate() -> dict:
    targets = SLOTargets()
    results: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targets": asdict(targets),
        "measurements": {},
        "pass_fail": {},
        "notes": [],
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1) Ingestion success/latency test
        csv_file = tmp_path / "telemetry.csv"
        csv_file.write_text("ts,val,extra\n1,10,100\n2,11,101\n3,9,99\n", encoding="utf-8")

        ingest_runs = 300
        ingest_success = 0
        ingest_latencies = []
        for _ in range(ingest_runs):
            start = time.perf_counter()
            ok, _, _ = DataReader.load_csv(str(csv_file))
            ingest_latencies.append((time.perf_counter() - start) * 1000.0)
            if ok:
                ingest_success += 1

        ingestion_success_rate = (ingest_success / ingest_runs) * 100.0
        results["measurements"]["ingestion"] = {
            "runs": ingest_runs,
            "success_rate_pct": round(ingestion_success_rate, 3),
            "latency_avg_ms": round(mean(ingest_latencies), 3),
            "latency_p95_ms": round(sorted(ingest_latencies)[int(0.95 * len(ingest_latencies)) - 1], 3),
        }
        results["pass_fail"]["ingestion_success_rate"] = ingestion_success_rate >= targets.ingestion_success_rate_pct

        # 2) Alert latency test (policy + persistence)
        db_path = str(tmp_path / "alerts.db")
        _init_alert_schema(db_path)

        def _noop_dispatcher(**kwargs):
            return None

        policy = AlertPolicyManager(
            db_path=db_path,
            config_loader=lambda: _make_config(
                routes={"info": ["webhook"], "warning": ["webhook"], "critical": ["webhook"]},
                channels={"webhook": {"enabled": True, "url": "http://example.invalid"}},
            ),
            dispatchers={"webhook": _noop_dispatcher},
        )

        alert_runs = 300
        alert_latencies = []
        for i in range(alert_runs):
            payload = {"model_name": "gate_model", "reconstruction_error": float(i % 3), "threshold": 1.0}
            start = time.perf_counter()
            policy.process_alert("warning", f"Synthetic alert {i}", payload)
            alert_latencies.append((time.perf_counter() - start) * 1000.0)

        alert_p95 = sorted(alert_latencies)[int(0.95 * len(alert_latencies)) - 1]
        results["measurements"]["alerting"] = {
            "runs": alert_runs,
            "latency_avg_ms": round(mean(alert_latencies), 3),
            "latency_p95_ms": round(alert_p95, 3),
        }
        results["pass_fail"]["alert_latency_p95"] = alert_p95 <= targets.alert_latency_p95_ms

        # 3) False positive bound estimate on synthetic-normal signals
        # We treat score < 0.2 as anomaly trigger for synthetic "normal" data.
        normal_errors = [0.05, 0.1, 0.2, 0.15, 0.08, 0.12, 0.18, 0.09, 0.11, 0.07] * 50
        threshold = 0.5
        predicted_anomalies = 0
        for err in normal_errors:
            health_score = compute_health_score(err, threshold)
            if health_score < 0.2:
                predicted_anomalies += 1
        fpr = (predicted_anomalies / len(normal_errors)) * 100.0
        results["measurements"]["false_positive"] = {
            "samples": len(normal_errors),
            "false_positive_rate_pct": round(fpr, 3),
            "threshold_reference": threshold,
        }
        results["pass_fail"]["false_positive_rate"] = fpr <= targets.false_positive_rate_pct

        # 4) Failure injection scenarios
        failed_ingest_ok, _, _ = DataReader.load_csv(str(tmp_path / "missing.csv"))
        results["measurements"]["failure_injection"] = {
            "missing_file_ingest_returns_false": failed_ingest_ok is False,
        }
        results["pass_fail"]["failure_injection_missing_file"] = failed_ingest_ok is False

        def _failing_dispatcher(**kwargs):
            raise RuntimeError("simulated webhook outage")

        policy_fail = AlertPolicyManager(
            db_path=db_path,
            config_loader=lambda: _make_config(
                routes={"critical": ["webhook"]},
                channels={"webhook": {"enabled": True, "url": "http://example.invalid"}},
            ),
            dispatchers={"webhook": _failing_dispatcher},
        )
        fail_result = policy_fail.process_alert("critical", "Injected route failure", {"model_name": "gate_model"})
        results["measurements"]["failure_injection"]["route_failure_handled"] = bool(fail_result.get("incident_id"))
        results["pass_fail"]["failure_injection_route_failure"] = bool(fail_result.get("incident_id"))

    results["overall_pass"] = all(bool(v) for v in results["pass_fail"].values())
    return results


def main() -> int:
    report = run_gate()
    output_dir = Path("reports") / "production"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "readiness_report.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {output_file}")
    return 0 if report.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

