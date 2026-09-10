from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import MAX_PRICE_AGE_DAYS, PROCESSED_DATA_DIR
from src.calendars import reference_sessions
from src.data_loader import ticker_to_filename
from src.features import FEATURE_COLUMNS, add_features
from src.targets import (
    ALPHA_TARGET_COLUMN,
    FUTURE_RETURN_COLUMN,
    OUTPERFORM_TARGET_COLUMN,
    SAMPLE_INDEX_NAME,
    TICKER_COLUMN,
    TARGET_DATE_COLUMN,
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
    return PROCESSED_DATA_DIR / f"stock_{ticker_to_filename(ticker)}_samples.parquet"


def all_samples_path() -> Path:
    return PROCESSED_DATA_DIR / ALL_SAMPLES_FILENAME


def build_ticker_samples(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    forecast_days: int,
    *,
    snapshot_date: str | pd.Timestamp | None = None,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
    reference_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(snapshot_date) if snapshot_date is not None else prices.index.max()
    benchmark = benchmark_prices.loc[benchmark_prices.index <= cutoff]
    prices = prices.loc[prices.index <= cutoff]
    dates = (
        pd.date_range(prices.index.min(), cutoff, freq="ME")
        if not prices.empty else pd.DatetimeIndex([])
    )
    if reference_dates is None:
        reference_dates = (
            reference_sessions(prices.index.min(), cutoff, forecast_days)
            if not prices.empty else pd.DatetimeIndex([])
        )
    samples = build_monthly_samples(
        add_features(prices), ticker, as_of_dates=dates,
        max_price_age_days=max_price_age_days,
    )
    with_targets = add_targets(
        samples, forecast_days, prices=prices, reference_dates=reference_dates,
        max_price_age_days=max_price_age_days, observation_end=cutoff,
    )
    with_alpha = add_benchmark_targets(
        with_targets, benchmark, max_price_age_days,
        reference_dates=reference_dates, observation_end=cutoff,
    )
    with_alpha["label_status"] = "missing"
    with_alpha.loc[with_alpha[TARGET_DATE_COLUMN] > cutoff, "label_status"] = "pending"
    with_alpha.loc[with_alpha[ALPHA_TARGET_COLUMN].notna(), "label_status"] = "observed"
    with_alpha["snapshot_date"] = cutoff
    # Eligibility depends only on the information available at as_of.
    return with_alpha.dropna(subset=[*FEATURE_COLUMNS, "Close", "price_date"])


def save_ticker_samples(samples: pd.DataFrame, ticker: str, output_dir: Path | None = None) -> Path:
    directory = PROCESSED_DATA_DIR if output_dir is None else output_dir
    directory.mkdir(parents=True, exist_ok=True)
    # ALL is a real ticker; ALL_samples collides with all_samples on Windows.
    output_path = directory / f"stock_{ticker_to_filename(ticker)}_samples.parquet"
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
        return pd.DataFrame(
            columns=[*REQUIRED_COMBINED_COLUMNS, *FEATURE_COLUMNS, TARGET_DATE_COLUMN],
            index=index,
        )

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


def save_combined_samples(samples: pd.DataFrame, output_dir: Path | None = None) -> Path:
    directory = PROCESSED_DATA_DIR if output_dir is None else output_dir
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / ALL_SAMPLES_FILENAME
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
