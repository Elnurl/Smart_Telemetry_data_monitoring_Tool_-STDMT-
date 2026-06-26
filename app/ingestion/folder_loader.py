from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ingestion.data_reader import DataReader


@dataclass
class LoadedDataset:
    dataframe: pd.DataFrame
    source_path: str
    source_mtime: float
    row_count: int
    timestamp_column: str


def find_latest_file(data_folder: Path, file_type: str = "CSV") -> Path | None:
    if not data_folder.exists():
        return None
    pattern = "*.csv" if file_type.upper() == "CSV" else "*.json"
    files = list(data_folder.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_latest_from_folder(data_folder: str | Path, file_type: str = "CSV") -> tuple[bool, str, LoadedDataset | None]:
    folder = Path(data_folder)
    latest = find_latest_file(folder, file_type)
    if latest is None:
        return False, f"No {file_type} files found in {folder}", None

    path = str(latest)
    if file_type.upper() == "JSON":
        ok, message, result = DataReader.load_json(path)
    else:
        ok, message, result = DataReader.load_csv(path)

    if not ok or result is None:
        return False, message, None

    return (
        True,
        message,
        LoadedDataset(
            dataframe=result.dataframe,
            source_path=path,
            source_mtime=os.path.getmtime(latest),
            row_count=len(result.dataframe),
            timestamp_column=result.timestamp_column,
        ),
    )


def timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).lower() in ("time", "timestamp", "datetime", "date", "ds"):
            return str(col)
    return None


def apply_dataset_mode(data: pd.DataFrame, mode: str, monitoring_window_rows: int = 500) -> pd.DataFrame:
    if data is None or len(data) == 0 or mode == "Full Latest File":
        return data

    ts_col = timestamp_column(data)
    if not ts_col:
        return data

    ts = pd.to_datetime(data[ts_col], errors="coerce")
    if not ts.notna().any():
        return data

    anchor = ts.max()
    if mode == "Daily":
        start = anchor - pd.Timedelta(days=1)
    elif mode == "Weekly":
        start = anchor - pd.Timedelta(days=7)
    elif mode == "Monthly":
        start = anchor - pd.Timedelta(days=30)
    else:
        return data

    filtered = data.loc[ts >= start].copy()
    if len(filtered) > 0:
        return filtered
    return data.tail(monitoring_window_rows).copy()


def numeric_features(df: pd.DataFrame, selected: list[str] | None = None) -> list[str]:
    cols = [str(c) for c in df.select_dtypes(include="number").columns]
    if selected:
        return [c for c in cols if c in selected]
    return cols
