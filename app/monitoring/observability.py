"""Lightweight observability: traces, metrics, health HTTP."""
from __future__ import annotations

import datetime
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("SecureAnomalyDetection")


class ObservabilityManager:
    """Lightweight observability manager: traces, metrics, and health state."""
    def __init__(self, service_name="telemetry_system"):
        self.service_name = service_name
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._summaries_sum = defaultdict(float)
        self._summaries_count = defaultdict(int)
        self._recent_errors = []
        self._http_server = None
        self._http_thread = None
    
    @staticmethod
    def _make_key(metric_name, labels=None):
        labels = labels or {}
        return metric_name, tuple(sorted((str(k), str(v)) for k, v in labels.items()))
    
    @staticmethod
    def _labels_to_text(labels_tuple):
        if not labels_tuple:
            return ""
        escaped = []
        for key, value in labels_tuple:
            safe_value = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            escaped.append(f'{key}="{safe_value}"')
        return "{" + ",".join(escaped) + "}"
    
    def inc_counter(self, metric_name, value=1.0, labels=None):
        key = self._make_key(metric_name, labels)
        with self._lock:
            self._counters[key] += float(value)
    
    def set_gauge(self, metric_name, value, labels=None):
        key = self._make_key(metric_name, labels)
        with self._lock:
            self._gauges[key] = float(value)
    
    def observe(self, metric_name, value, labels=None):
        key = self._make_key(metric_name, labels)
        with self._lock:
            self._summaries_sum[key] += float(value)
            self._summaries_count[key] += 1
    
    def start_span(self, operation, attributes=None):
        return {
            "trace_id": uuid.uuid4().hex,
            "span_id": uuid.uuid4().hex[:16],
            "operation": operation,
            "start_time": time.time(),
            "attributes": attributes or {}
        }
    
    def end_span(self, span, status="ok", error=None, attributes=None):
        if not span:
            return
        duration_s = max(0.0, time.time() - span["start_time"])
        op = span.get("operation", "unknown")
        
        labels = {"operation": op, "status": status}
        self.inc_counter("app_operations_total", 1, labels=labels)
        self.observe("app_operation_duration_seconds", duration_s, labels={"operation": op})
        
        event = {
            "event": "trace_span",
            "service": self.service_name,
            "trace_id": span.get("trace_id"),
            "span_id": span.get("span_id"),
            "operation": op,
            "status": status,
            "duration_ms": round(duration_s * 1000, 2),
            "ts_utc": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        merged_attrs = {}
        merged_attrs.update(span.get("attributes", {}))
        if attributes:
            merged_attrs.update(attributes)
        if merged_attrs:
            event["attributes"] = merged_attrs
        if error:
            event["error"] = str(error)
        
        if status != "ok":
            self.inc_counter("app_errors_total", 1, labels={"operation": op})
            with self._lock:
                self._recent_errors.append(time.time())
                if len(self._recent_errors) > 1000:
                    self._recent_errors = self._recent_errors[-1000:]
        
        logger.info("OBS_EVENT %s", json.dumps(event, ensure_ascii=False))
    
    @contextmanager
    def trace(self, operation, attributes=None):
        span = self.start_span(operation, attributes=attributes)
        try:
            yield span
            self.end_span(span, status="ok")
        except Exception as exc:
            self.end_span(span, status="error", error=exc)
            raise
    
    def health_payload(self):
        uptime_s = max(0.0, time.time() - self.started_at)
        now = time.time()
        with self._lock:
            recent_5m_errors = sum(1 for ts in self._recent_errors if now - ts <= 300)
        
        status = "ok"
        if recent_5m_errors >= 10:
            status = "degraded"
        
        return {
            "status": status,
            "service": self.service_name,
            "uptime_seconds": round(uptime_s, 2),
            "recent_errors_5m": recent_5m_errors,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"
        }
    
    def metrics_text(self):
        lines = []
        uptime = max(0.0, time.time() - self.started_at)
        health = self.health_payload()
        
        self.set_gauge("app_uptime_seconds", uptime)
        self.set_gauge("app_health_status", 1 if health["status"] == "ok" else 0)
        self.set_gauge("app_recent_errors_5m", health["recent_errors_5m"])
        
        lines.append("# HELP app_operations_total Total number of instrumented operations.")
        lines.append("# TYPE app_operations_total counter")
        lines.append("# HELP app_errors_total Total number of failed operations.")
        lines.append("# TYPE app_errors_total counter")
        lines.append("# HELP app_operation_duration_seconds_sum Sum of operation durations.")
        lines.append("# TYPE app_operation_duration_seconds_sum counter")
        lines.append("# HELP app_operation_duration_seconds_count Count of operation durations.")
        lines.append("# TYPE app_operation_duration_seconds_count counter")
        lines.append("# HELP app_uptime_seconds Process uptime in seconds.")
        lines.append("# TYPE app_uptime_seconds gauge")
        lines.append("# HELP app_health_status 1=healthy, 0=degraded.")
        lines.append("# TYPE app_health_status gauge")
        lines.append("# HELP app_recent_errors_5m Number of recent errors (5m window).")
        lines.append("# TYPE app_recent_errors_5m gauge")
        
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            sum_data = dict(self._summaries_sum)
            count_data = dict(self._summaries_count)
        
        for (name, labels), value in counters.items():
            lines.append(f"{name}{self._labels_to_text(labels)} {value}")
        
        for (name, labels), value in sum_data.items():
            lines.append(f"{name}_sum{self._labels_to_text(labels)} {value}")
        for (name, labels), value in count_data.items():
            lines.append(f"{name}_count{self._labels_to_text(labels)} {value}")
        
        for (name, labels), value in gauges.items():
            lines.append(f"{name}{self._labels_to_text(labels)} {value}")
        
        return "\n".join(lines) + "\n"
    
    def start_http_server(self, host="127.0.0.1", port=9108):
        """Expose /healthz and /metrics endpoints in background thread."""
        if self._http_server is not None:
            return
        
        manager = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/health", "/healthz"):
                    payload = manager.health_payload()
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                
                if self.path == "/metrics":
                    body = manager.metrics_text().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                
                self.send_response(404)
                self.end_headers()
            
            def log_message(self, format, *args):
                return
        
        self._http_server = ThreadingHTTPServer((host, int(port)), Handler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
        logger.info(f"Observability endpoint started at http://{host}:{port} (/healthz, /metrics)")


# Process-wide singleton (main.py starts HTTP when enabled).
OBSERVABILITY = ObservabilityManager(service_name="telemetry_system")
