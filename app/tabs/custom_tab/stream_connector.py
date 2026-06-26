"""MQTT / serial / OPC-UA stream ingestion for custom monitoring tabs (PyQt-free core)."""

from __future__ import annotations

import datetime
import json
import logging
import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import serial  # type: ignore
except Exception:
    serial = None

logger = logging.getLogger("STDMS.StreamConnector")

StatusCallback = Optional[Callable[[str], None]]


def parse_serial_line(
    line: str,
    *,
    delimiter: str = ",",
    field_names: Optional[List[str]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse one serial line into a telemetry record dict."""
    record: Dict[str, Any] = {"timestamp": timestamp or datetime.datetime.now().isoformat()}
    text = str(line).strip()
    if not text:
        return record

    try:
        if text.startswith("{") and text.endswith("}"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                record.update(payload)
            return record

        parts = [part.strip() for part in text.split(delimiter[:1] or ",")]
        names = [name.strip() for name in (field_names or []) if str(name).strip()]
        for idx, value in enumerate(parts):
            key = names[idx] if idx < len(names) else f"ch_{idx + 1}"
            try:
                record[key] = float(value)
            except Exception:
                record[key] = value
    except Exception:
        record["raw_line"] = text
    return record


class StreamRecordBuffer:
    """Thread-safe deque buffer for incoming stream records."""

    def __init__(self, maxlen: int = 2000):
        self._records: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, record: Any) -> None:
        if not isinstance(record, dict):
            return
        with self._lock:
            self._records.append(record)

    def consume_batch(self, max_records: int = 500) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        rows: List[dict] = []
        with self._lock:
            while self._records and len(rows) < max_records:
                rows.append(self._records.popleft())
        if not rows:
            return None, "No stream data received yet"
        return pd.DataFrame(rows), None


class StreamConnector:
    """Manage background stream workers (serial, MQTT, OPC-UA)."""

    def __init__(self, buffer: StreamRecordBuffer):
        self.buffer = buffer
        self.stop_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.mqtt_client: Any = None
        self.serial_connection: Any = None

    def stop(self) -> None:
        try:
            if self.stop_event is not None:
                self.stop_event.set()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
            self.stop_event = None
        except Exception as exc:
            logger.warning("Error stopping stream thread: %s", exc)

        try:
            if self.mqtt_client is not None:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
        except Exception:
            pass
        self.mqtt_client = None

        try:
            if self.serial_connection is not None:
                self.serial_connection.close()
        except Exception:
            pass
        self.serial_connection = None

    def start(self, config: dict, *, status_callback: StatusCallback = None) -> Tuple[bool, Optional[str]]:
        if bool(config.get("active_device_monitoring_enabled", False)):
            return self._start_serial(config, status_callback=status_callback)

        mode = config.get("input_mode", "CSV Polling")
        if mode == "MQTT Stream":
            return self._start_mqtt(config, status_callback=status_callback)
        if mode == "OPC-UA Stream":
            return self._start_opcua(config)
        return True, None

    def _start_serial(self, config: dict, *, status_callback: StatusCallback = None) -> Tuple[bool, Optional[str]]:
        port = str(config.get("serial_port", "")).strip()
        baudrate = int(config.get("serial_baudrate", 115200))
        delimiter = str(config.get("serial_delimiter", ",") or ",")[:1]
        field_names = [f.strip() for f in config.get("serial_fields", []) if str(f).strip()]
        if not port:
            return False, "Serial port is required for active device monitoring"
        if serial is None:
            return False, "pyserial not installed (pip install pyserial)"

        self.stop_event = threading.Event()
        buffer = self.buffer

        def _worker() -> None:
            ser = None
            try:
                ser = serial.Serial(port=port, baudrate=baudrate, timeout=1.0)
                self.serial_connection = ser
                if status_callback:
                    status_callback(f"Serial connected: {port} @ {baudrate}")
                while not self.stop_event.is_set():
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    record = parse_serial_line(line, delimiter=delimiter, field_names=field_names)
                    buffer.append(record)
            except Exception as exc:
                logger.error("Serial stream error: %s", exc)
            finally:
                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()
        return True, None

    def _start_mqtt(self, config: dict, *, status_callback: StatusCallback = None) -> Tuple[bool, Optional[str]]:
        broker = str(config.get("mqtt_broker", "")).strip()
        topic = str(config.get("mqtt_topic", "")).strip()
        port = int(config.get("mqtt_port", 1883))
        if not broker or not topic:
            return False, "MQTT broker/topic is required"

        try:
            import paho.mqtt.client as mqtt
        except Exception:
            return False, "paho-mqtt not installed (pip install paho-mqtt)"

        self.stop_event = threading.Event()
        buffer = self.buffer

        def _on_connect(client, _userdata, _flags, rc: int) -> None:
            if rc == 0:
                client.subscribe(topic)
            if not status_callback:
                return
            if rc == 0:
                status_callback(f"MQTT connected: {broker}:{port} topic={topic}")
            else:
                status_callback(f"MQTT connect failed rc={rc}")

        def _on_message(_client, _userdata, msg) -> None:
            try:
                payload = msg.payload.decode("utf-8", errors="ignore")
                data = json.loads(payload)
                if isinstance(data, list):
                    for rec in data:
                        buffer.append(rec)
                elif isinstance(data, dict):
                    buffer.append(data)
            except Exception as exc:
                logger.warning("MQTT payload parse error: %s", exc)

        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message
        client.connect(broker, port, keepalive=30)
        client.loop_start()
        self.mqtt_client = client
        return True, None

    def _start_opcua(self, config: dict) -> Tuple[bool, Optional[str]]:
        endpoint = str(config.get("opcua_endpoint", "")).strip()
        nodes_text = str(config.get("opcua_nodes", "")).strip()
        if not endpoint or not nodes_text:
            return False, "OPC-UA endpoint and nodes are required"

        try:
            from opcua import Client as OpcUaClient  # type: ignore
        except Exception:
            return False, "opcua package not installed (pip install opcua)"

        node_ids = [node.strip() for node in nodes_text.split(",") if node.strip()]
        if not node_ids:
            return False, "No valid OPC-UA node IDs provided"

        poll_seconds = max(1.0, float(config.get("interval_ms", 300000)) / 1000.0)
        self.stop_event = threading.Event()
        buffer = self.buffer

        def _worker() -> None:
            client = OpcUaClient(endpoint)
            try:
                client.connect()
                while not self.stop_event.is_set():
                    record = {"timestamp": datetime.datetime.now().isoformat()}
                    for node_id in node_ids:
                        try:
                            record[node_id] = client.get_node(node_id).get_value()
                        except Exception:
                            record[node_id] = np.nan
                    buffer.append(record)
                    self.stop_event.wait(timeout=poll_seconds)
            except Exception as exc:
                logger.error("OPC-UA stream error: %s", exc)
            finally:
                try:
                    client.disconnect()
                except Exception:
                    pass

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()
        return True, None
