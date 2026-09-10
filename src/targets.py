from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAX_PRICE_AGE_DAYS
from src.calendars import reference_sessions


FUTURE_RETURN_COLUMN = "future_return_12m"
BENCHMARK_RETURN_COLUMN = "benchmark_return_12m"
ALPHA_TARGET_COLUMN = "alpha_12m"
OUTPERFORM_TARGET_COLUMN = "future_outperform_12m"
TARGET_DATE_COLUMN = "future_target_date"
TICKER_COLUMN = "ticker"
SAMPLE_INDEX_NAME = "as_of"
PRICE_DATE_COLUMN = "price_date"

TARGET_COLUMNS = [
    FUTURE_RETURN_COLUMN, BENCHMARK_RETURN_COLUMN, ALPHA_TARGET_COLUMN,
    OUTPERFORM_TARGET_COLUMN, "future_positive_12m", "future_plus_10_12m",
    "future_plus_20_12m", "future_minus_10_12m", "future_minus_20_12m",
]


def _binary_target(values: pd.Series, condition: pd.Series) -> pd.Series:
    target = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.notna()
    target.loc[valid] = condition.loc[valid].astype(int)
    return target


def _require_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if df.index.tz is not None or df.index.hasnans or not df.index.is_unique:
        raise ValueError("Expected unique, timezone-naive dates without NaT.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Expected dates sorted in ascending order.")


def prices_at_or_before(
    close: pd.Series,
    lookup_dates: pd.Series,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
) -> pd.DataFrame:
    """Match past prices with a bounded age and retain their actual dates."""
    if max_price_age_days < 0:
        raise ValueError("max_price_age_days must not be negative.")
    _require_datetime_index(close.to_frame())
    source = pd.DataFrame({
        "price_date": close.index.to_numpy(dtype="datetime64[ns]"),
        "price": pd.to_numeric(close, errors="coerce").to_numpy(dtype=float),
    })
    source = source.loc[np.isfinite(source.price) & (source.price > 0)]
    requests = pd.DataFrame({
        "lookup_date": pd.to_datetime(lookup_dates).to_numpy(dtype="datetime64[ns]"),
        "position": np.arange(len(lookup_dates)),
    }).dropna(subset=["lookup_date"]).sort_values("lookup_date")
    result = pd.DataFrame({
        "price": np.full(len(lookup_dates), np.nan),
        "price_date": np.full(len(lookup_dates), np.datetime64("NaT", "ns")),
    }, index=lookup_dates.index)
    if source.empty or requests.empty:
        return result
    matched = pd.merge_asof(
        requests, source, left_on="lookup_date", right_on="price_date",
        direction="backward", tolerance=pd.Timedelta(days=max_price_age_days),
    )
    result.iloc[matched.position.to_numpy(), 0] = matched.price.to_numpy()
    result.iloc[matched.position.to_numpy(), 1] = matched.price_date.to_numpy()
    return result


def add_targets(
    df: pd.DataFrame,
    forecast_days: int = 252,
    *,
    prices: pd.DataFrame | None = None,
    reference_dates: pd.DatetimeIndex | None = None,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
    observation_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Measure returns to the Nth reference session strictly after each as_of.

    The production pipeline supplies one independent exchange calendar.
    Planned target dates exist even when the outcome has not matured yet.
    """
    _require_datetime_index(df)
    if "Close" not in df.columns:
        raise ValueError("Expected a Close column.")
    if not isinstance(forecast_days, int) or forecast_days <= 0:
        raise ValueError("forecast_days must be a positive integer.")
    prices = df if prices is None else prices
    _require_datetime_index(prices)
    calendar = reference_dates
    if calendar is None:
        calendar = (
            reference_sessions(df.index.min(), df.index.max(), forecast_days)
            if not df.empty else pd.DatetimeIndex([])
        )
    _require_datetime_index(pd.DataFrame(index=calendar))
    positions = calendar.searchsorted(df.index, side="right") + forecast_days - 1
    available = positions < len(calendar)
    if not available.all():
        raise ValueError("Reference calendar does not cover every target date.")
    future_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    future_dates.iloc[np.flatnonzero(available)] = calendar[positions[available]].to_numpy()
    cutoff = prices.index.max() if observation_end is None else pd.Timestamp(observation_end)
    future = prices_at_or_before(
        prices["Close"], future_dates.where(future_dates <= cutoff), max_price_age_days,
    )

    targeted = df.copy()
    targeted["forecast_days"] = forecast_days
    targeted[TARGET_DATE_COLUMN] = future_dates
    targeted["future_price_date"] = future.price_date
    targeted["future_price_12m"] = future.price
    close = pd.to_numeric(targeted["Close"], errors="coerce")
    targeted[FUTURE_RETURN_COLUMN] = (future.price / close - 1).where(close > 0)
    future_return = targeted[FUTURE_RETURN_COLUMN]
    conditions = {
        "future_positive_12m": future_return > 0,
        "future_plus_10_12m": future_return > 0.10,
        "future_plus_20_12m": future_return > 0.20,
        "future_minus_10_12m": future_return < -0.10,
        "future_minus_20_12m": future_return < -0.20,
    }
    for column, condition in conditions.items():
        targeted[column] = _binary_target(future_return, condition)
    return targeted


def add_benchmark_targets(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
    *,
    reference_dates: pd.DatetimeIndex | None = None,
    observation_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Alpha is the simple difference of adjusted returns, not factor alpha."""
    _require_datetime_index(df)
    _require_datetime_index(benchmark_df)
    if FUTURE_RETURN_COLUMN not in df.columns or TARGET_DATE_COLUMN not in df.columns:
        raise ValueError("Run add_targets before add_benchmark_targets.")
    if "Close" not in benchmark_df.columns:
        raise ValueError("Expected benchmark data with a Close column.")
    targeted = df.copy()
    if reference_dates is None:
        reference_dates = (
            reference_sessions(df.index.min(), max(df.index.max(), df[TARGET_DATE_COLUMN].max()))
            if not df.empty else pd.DatetimeIndex([])
        )
    cutoff = benchmark_df.index.max() if observation_end is None else pd.Timestamp(observation_end)
    _require_datetime_index(pd.DataFrame(index=reference_dates))
    start = _benchmark_at_expected_session(
        benchmark_df.Close, pd.Series(df.index, index=df.index), reference_dates,
        max_price_age_days,
    )
    future = _benchmark_at_expected_session(
        benchmark_df.Close, df[TARGET_DATE_COLUMN].where(df[TARGET_DATE_COLUMN] <= cutoff),
        reference_dates, max_price_age_days,
    )
    targeted["benchmark_price_date"] = start.price_date
    targeted["benchmark_future_price_date"] = future.price_date
    targeted[BENCHMARK_RETURN_COLUMN] = future.price / start.price - 1
    targeted[ALPHA_TARGET_COLUMN] = targeted[FUTURE_RETURN_COLUMN] - targeted[BENCHMARK_RETURN_COLUMN]
    targeted[OUTPERFORM_TARGET_COLUMN] = _binary_target(
        targeted[ALPHA_TARGET_COLUMN], targeted[ALPHA_TARGET_COLUMN] > 0,
    )
    return targeted


def _benchmark_at_expected_session(
    close: pd.Series,
    requested: pd.Series,
    sessions: pd.DatetimeIndex,
    max_price_age_days: int,
) -> pd.DataFrame:
    """Use the expected preceding session, not the preceding available quote."""
    if max_price_age_days < 0:
        raise ValueError("max_price_age_days must not be negative.")
    positions = sessions.searchsorted(requested, side="right") - 1
    valid = requested.notna().to_numpy() & (positions >= 0) & (positions < len(sessions))
    expected = pd.Series(pd.NaT, index=requested.index, dtype="datetime64[ns]")
    expected.iloc[np.flatnonzero(valid)] = sessions[positions[valid]].to_numpy()
    values = pd.Series(
        pd.to_numeric(close.reindex(pd.DatetimeIndex(expected)), errors="coerce").to_numpy(),
        index=requested.index, dtype=float,
    )
    age = requested - expected
    valid_price = np.isfinite(values) & values.gt(0) & age.between(
        pd.Timedelta(0), pd.Timedelta(days=max_price_age_days),
    )
    return pd.DataFrame({"price": values.where(valid_price), "price_date": expected.where(valid_price)})


def build_monthly_samples(
    df: pd.DataFrame,
    ticker: str,
    *,
    as_of_dates: pd.DatetimeIndex | None = None,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
) -> pd.DataFrame:
    """Select fixed month ends before checking feature or label availability.

    as_of denotes a decision after all exchanges have closed on that date.
    A missing feature on the selected row never selects an earlier row.
    """
    _require_datetime_index(df)
    if not ticker.strip():
        raise ValueError("ticker must not be empty.")
    if "Close" not in df.columns:
        raise ValueError("Expected a Close column.")
    if as_of_dates is None:
        as_of_dates = (
            pd.date_range(df.index.min(), df.index.max(), freq="ME")
            if not df.empty else pd.DatetimeIndex([])
        )
    _require_datetime_index(pd.DataFrame(index=as_of_dates))
    if not as_of_dates.is_month_end.all():
        raise ValueError("Decision dates must be calendar month ends.")
    if max_price_age_days < 0:
        raise ValueError("max_price_age_days must not be negative.")
    sampled = df.drop(columns=[TICKER_COLUMN], errors="ignore").copy()
    sampled[PRICE_DATE_COLUMN] = sampled.index
    sampled = sampled.reindex(
        as_of_dates, method="ffill", tolerance=pd.Timedelta(days=max_price_age_days),
    )
    sampled.insert(0, TICKER_COLUMN, ticker)
    sampled.index.name = SAMPLE_INDEX_NAME
    return sampled
