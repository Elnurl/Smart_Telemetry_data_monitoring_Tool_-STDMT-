from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

TIMESTAMP_COLUMN_NAMES = frozenset({"time", "timestamp", "datetime", "date", "ds", "ts"})
_GENERIC_FEATURE_NAMES = frozenset({"val", "value", "v", "extra", "x", "y", "metric"})


@dataclass
class NormalizedDataResult:
    dataframe: pd.DataFrame
    timestamp_column: str
    original_columns: list[str]
    rename_map: Dict[str, str]


class DataReader:
    """Reusable ingestion reader for CSV/JSON normalization."""

    @staticmethod
    def _detect_timestamp_column(columns) -> str:
        for col in columns:
            if str(col).strip().lower() in TIMESTAMP_COLUMN_NAMES:
                return str(col)
        return ""

    @staticmethod
    def _looks_like_data_used_as_header(columns: list[str]) -> bool:
        for raw in columns:
            name = str(raw).strip()
            compact = name.replace(".", "", 1).replace("-", "", 1)
            if compact.isdigit():
                return True
            if len(name) >= 10 and name[:4].isdigit() and name[4] in "-T":
                return True
        return False

    @classmethod
    def _is_generic_schema(cls, columns: list[str]) -> bool:
        if cls._looks_like_data_used_as_header(columns):
            return True
        names = [str(c).strip().lower() for c in columns]
        features = [n for n in names if n not in TIMESTAMP_COLUMN_NAMES]
        if not features:
            return True
        return all(
            n in _GENERIC_FEATURE_NAMES or n.startswith("unnamed") or n.startswith("value")
            for n in features
        )

    @classmethod
    def _normalize_dataframe(cls, df: pd.DataFrame) -> NormalizedDataResult:
        original_columns = [str(c) for c in df.columns]
        if len(original_columns) < 2:
            return NormalizedDataResult(
                dataframe=df,
                timestamp_column=cls._detect_timestamp_column(original_columns),
                original_columns=original_columns,
                rename_map={},
            )

        if not cls._is_generic_schema(original_columns):
            ts = cls._detect_timestamp_column(original_columns)
            if not ts:
                sample = df[original_columns[0]].head(min(20, len(df)))
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    ts = original_columns[0]
            return NormalizedDataResult(
                dataframe=df,
                timestamp_column=ts,
                original_columns=original_columns,
                rename_map={},
            )

        rename_map = {original_columns[0]: "time", original_columns[1]: "value"}
        for i, col in enumerate(original_columns[2:], start=2):
            rename_map[col] = f"value{i}"
        normalized = df.rename(columns=rename_map)
        return NormalizedDataResult(
            dataframe=normalized,
            timestamp_column="time",
            original_columns=original_columns,
            rename_map=rename_map,
        )

    @classmethod
    def load_csv(cls, filepath: str) -> Tuple[bool, str, NormalizedDataResult | None]:
        try:
            df = pd.read_csv(filepath)
            result = cls._normalize_dataframe(df)
            if result.rename_map:
                message = "Data loaded successfully and column names standardized to ['time', 'value']"
            else:
                message = "Data loaded successfully"
            return True, message, result
        except Exception as e:
            return False, f"Error loading data: {str(e)}", None

    @classmethod
    def load_json(cls, filepath: str) -> Tuple[bool, str, NormalizedDataResult | None]:
        try:
            df = pd.read_json(filepath)
            result = cls._normalize_dataframe(df)
            return True, "Data loaded successfully and column names standardized to ['time', 'value']", result
        except Exception as e:
            return False, f"Error loading data: {str(e)}", None
