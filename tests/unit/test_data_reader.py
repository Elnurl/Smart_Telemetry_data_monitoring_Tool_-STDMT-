from pathlib import Path

from app.ingestion.data_reader import DataReader


def test_data_reader_preserves_satellite_channel_names(tmp_path: Path):
    csv_file = tmp_path / "sda_live.csv"
    csv_file.write_text(
        "timestamp,battery_voltage,solar_current,soc\n"
        "2026-01-01 00:00:00,27.4,6.2,81.0\n",
        encoding="utf-8",
    )
    ok, message, result = DataReader.load_csv(str(csv_file))
    assert ok is True
    assert result is not None
    assert list(result.dataframe.columns) == ["timestamp", "battery_voltage", "solar_current", "soc"]
    assert result.timestamp_column == "timestamp"
    assert "standardized" not in message


def test_data_reader_load_csv_normalizes_columns(tmp_path: Path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("ts,val,extra\n1,10,100\n2,11,101\n", encoding="utf-8")

    ok, message, result = DataReader.load_csv(str(csv_file))
    assert ok is True
    assert "standardized" in message
    assert result is not None
    assert list(result.dataframe.columns) == ["time", "value", "value2"]
    assert result.timestamp_column == "time"


def test_data_reader_load_json_normalizes_columns(tmp_path: Path):
    json_file = tmp_path / "sample.json"
    json_file.write_text('[{"ts":"2024-01-01","val":10}]', encoding="utf-8")

    ok, _, result = DataReader.load_json(str(json_file))
    assert ok is True
    assert result is not None
    assert list(result.dataframe.columns) == ["time", "value"]

