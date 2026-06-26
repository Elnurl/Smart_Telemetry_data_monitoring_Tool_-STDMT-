from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class NormalizedDataResult:
    dataframe: pd.DataFrame
    timestamp_column: str
    original_columns: list[str]
    rename_map: Dict[str, str]


class DataReader:
    """Reusable ingestion reader for CSV/JSON normalization."""

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> NormalizedDataResult:
        if len(df.columns) < 2:
            return NormalizedDataResult(
                dataframe=df,
                timestamp_column="",
                original_columns=list(df.columns),
                rename_map={},
            )

        original_columns = df.columns.tolist()
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
            return True, "Data loaded successfully and column names standardized to ['time', 'value']", result
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

