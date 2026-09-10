from __future__ import annotations

import pandas as pd

from src.features import FEATURE_COLUMNS


def latest_complete_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    complete = df.dropna(subset=FEATURE_COLUMNS)
    if complete.empty:
        raise ValueError("No complete feature row available.")

    return complete.tail(1)


def format_return(value: float) -> str:
    return f"{value:+.1%}"
