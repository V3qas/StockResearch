from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data_loader import ticker_to_filename
from src.features import add_features
from src.targets import (
    ALPHA_TARGET_COLUMN,
    FUTURE_RETURN_COLUMN,
    OUTPERFORM_TARGET_COLUMN,
    SAMPLE_INDEX_NAME,
    TICKER_COLUMN,
    add_benchmark_targets,
    add_targets,
    build_monthly_samples,
)


ALL_SAMPLES_FILENAME = "all_samples.parquet"
REQUIRED_COMBINED_COLUMNS = [
    TICKER_COLUMN,
    FUTURE_RETURN_COLUMN,
    ALPHA_TARGET_COLUMN,
    OUTPERFORM_TARGET_COLUMN,
]


def ticker_samples_path(ticker: str) -> Path:
    return PROCESSED_DATA_DIR / f"{ticker_to_filename(ticker)}_samples.parquet"


def all_samples_path() -> Path:
    return PROCESSED_DATA_DIR / ALL_SAMPLES_FILENAME


def build_ticker_samples(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    forecast_days: int,
) -> pd.DataFrame:
    with_features = add_features(prices)
    with_targets = add_targets(with_features, forecast_days=forecast_days)
    with_alpha = add_benchmark_targets(with_targets, benchmark_prices)
    return build_monthly_samples(with_alpha, ticker=ticker)


def save_ticker_samples(samples: pd.DataFrame, ticker: str) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ticker_samples_path(ticker)
    samples.to_parquet(output_path)
    return output_path


def combine_ticker_samples(sample_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for samples in sample_frames:
        if samples.empty:
            continue
        _validate_samples_for_combining(samples)

        normalized = samples.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized.index.name = SAMPLE_INDEX_NAME
        normalized[TICKER_COLUMN] = normalized[TICKER_COLUMN].astype(str)
        frames.append(normalized)

    if not frames:
        index = pd.DatetimeIndex([], name=SAMPLE_INDEX_NAME)
        return pd.DataFrame(columns=REQUIRED_COMBINED_COLUMNS, index=index)

    combined = pd.concat(frames, axis=0)
    combined = combined.reset_index().sort_values(
        [SAMPLE_INDEX_NAME, TICKER_COLUMN],
        kind="stable",
    )
    if combined.duplicated([SAMPLE_INDEX_NAME, TICKER_COLUMN]).any():
        raise ValueError("Duplicate sample rows found for the same as_of/ticker pair.")

    combined = combined.set_index(SAMPLE_INDEX_NAME)
    combined.index = pd.DatetimeIndex(combined.index, name=SAMPLE_INDEX_NAME)
    return combined


def save_combined_samples(samples: pd.DataFrame) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = all_samples_path()
    samples.to_parquet(output_path)
    return output_path


def _validate_samples_for_combining(samples: pd.DataFrame) -> None:
    if not isinstance(samples.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")

    missing_columns = [
        column for column in REQUIRED_COMBINED_COLUMNS if column not in samples.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing combined sample columns: {missing_columns}")
