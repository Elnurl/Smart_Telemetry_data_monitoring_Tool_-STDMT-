"""Unit tests for stream connector buffer and parsing."""

import pandas as pd
import pytest

from app.tabs.custom_tab.stream_connector import (
    StreamConnector,
    StreamRecordBuffer,
    parse_serial_line,
)


def test_parse_serial_line_json():
    record = parse_serial_line('{"temp": 42.5, "unit": "C"}', timestamp="2026-01-01T00:00:00")
    assert record["timestamp"] == "2026-01-01T00:00:00"
    assert record["temp"] == 42.5
    assert record["unit"] == "C"


def test_parse_serial_line_csv_with_field_names():
    record = parse_serial_line(
        "1.0,2.5,hello",
        delimiter=",",
        field_names=["a", "b", "c"],
        timestamp="t0",
    )
    assert record["a"] == 1.0
    assert record["b"] == 2.5
    assert record["c"] == "hello"


def test_parse_serial_line_csv_default_channel_names():
    record = parse_serial_line("3.14,7", delimiter=",", timestamp="t0")
    assert record["ch_1"] == 3.14
    assert record["ch_2"] == 7.0


def test_stream_record_buffer_append_and_consume():
    buf = StreamRecordBuffer(maxlen=10)
    buf.append({"v": 1})
    buf.append({"v": 2})
    buf.append("not a dict")  # ignored

    df, err = buf.consume_batch(max_records=5)
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df["v"]) == [1, 2]

    df2, err2 = buf.consume_batch()
    assert df2 is None
    assert err2 == "No stream data received yet"


def test_stream_connector_start_serial_missing_port():
    conn = StreamConnector(StreamRecordBuffer())
    ok, err = conn.start({"active_device_monitoring_enabled": True, "serial_port": ""})
    assert ok is False
    assert "Serial port" in (err or "")


def test_stream_connector_start_mqtt_missing_broker():
    conn = StreamConnector(StreamRecordBuffer())
    ok, err = conn.start({"input_mode": "MQTT Stream", "mqtt_broker": "", "mqtt_topic": "t"})
    assert ok is False
    assert "MQTT broker" in (err or "")


def test_stream_connector_csv_polling_mode_noop():
    conn = StreamConnector(StreamRecordBuffer())
    ok, err = conn.start({"input_mode": "CSV Polling"})
    assert ok is True
    assert err is None
