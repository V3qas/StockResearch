from __future__ import annotations

import numpy as np
import pandas as pd


FUTURE_RETURN_COLUMN = "future_return_12m"
BENCHMARK_RETURN_COLUMN = "benchmark_return_12m"
ALPHA_TARGET_COLUMN = "alpha_12m"
OUTPERFORM_TARGET_COLUMN = "future_outperform_12m"
TARGET_DATE_COLUMN = "future_target_date"

TARGET_COLUMNS = [
    FUTURE_RETURN_COLUMN,
    BENCHMARK_RETURN_COLUMN,
    ALPHA_TARGET_COLUMN,
    OUTPERFORM_TARGET_COLUMN,
    "future_positive_12m",
    "future_plus_10_12m",
    "future_plus_20_12m",
    "future_minus_10_12m",
    "future_minus_20_12m",
]


def _binary_target(values: pd.Series, condition: pd.Series) -> pd.Series:
    target = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.notna()
    target.loc[valid] = condition.loc[valid].astype(int)
    return target


def _require_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")


def _close_at_or_before(
    close: pd.Series,
    lookup_dates: pd.Series,
) -> pd.Series:
    benchmark_dates = pd.Series(pd.to_datetime(close.index), dtype="datetime64[ns]")
    benchmark = pd.DataFrame(
        {
            "lookup_date": benchmark_dates,
            "benchmark_close": pd.to_numeric(close, errors="coerce").to_numpy(),
        }
    ).dropna()
    benchmark = benchmark.sort_values("lookup_date")

    requests = pd.DataFrame(
        {
            "lookup_date": pd.Series(
                pd.to_datetime(lookup_dates, errors="coerce"),
                dtype="datetime64[ns]",
            ).to_numpy(),
            "position": np.arange(len(lookup_dates)),
        },
        index=lookup_dates.index,
    )
    result = pd.Series(np.nan, index=lookup_dates.index, dtype="float64")
    valid_requests = requests.dropna(subset=["lookup_date"]).sort_values("lookup_date")
    if benchmark.empty or valid_requests.empty:
        return result

    matched = pd.merge_asof(
        valid_requests,
        benchmark,
        on="lookup_date",
        direction="backward",
    )
    result.iloc[matched["position"].to_numpy()] = matched[
        "benchmark_close"
    ].to_numpy()

    return result


def add_targets(
    df: pd.DataFrame,
    forecast_days: int = 252,
) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Expected a Close column.")
    if forecast_days <= 0:
        raise ValueError("forecast_days must be positive.")

    targeted = df.copy()
    close = pd.to_numeric(targeted["Close"], errors="coerce")
    future_price = close.shift(-forecast_days)
    future_date = pd.Series(targeted.index, index=targeted.index).shift(-forecast_days)

    targeted["future_price_12m"] = future_price
    targeted[TARGET_DATE_COLUMN] = future_date
    targeted[FUTURE_RETURN_COLUMN] = future_price / close - 1

    future_return = targeted[FUTURE_RETURN_COLUMN]
    targeted["future_positive_12m"] = _binary_target(
        future_return,
        future_return > 0,
    )
    targeted["future_plus_10_12m"] = _binary_target(
        future_return,
        future_return > 0.10,
    )
    targeted["future_plus_20_12m"] = _binary_target(
        future_return,
        future_return > 0.20,
    )
    targeted["future_minus_10_12m"] = _binary_target(
        future_return,
        future_return < -0.10,
    )
    targeted["future_minus_20_12m"] = _binary_target(
        future_return,
        future_return < -0.20,
    )

    return targeted


def add_benchmark_targets(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    _require_datetime_index(df)
    _require_datetime_index(benchmark_df)
    if FUTURE_RETURN_COLUMN not in df.columns or TARGET_DATE_COLUMN not in df.columns:
        raise ValueError("Run add_targets before add_benchmark_targets.")
    if "Close" not in benchmark_df.columns:
        raise ValueError("Expected benchmark data with a Close column.")

    targeted = df.copy()
    benchmark = benchmark_df.sort_index()
    benchmark_close = pd.to_numeric(benchmark["Close"], errors="coerce").dropna()

    as_of_dates = pd.Series(targeted.index, index=targeted.index)
    future_dates = pd.Series(
        pd.to_datetime(targeted[TARGET_DATE_COLUMN], errors="coerce"),
        index=targeted.index,
    )

    benchmark_close_as_of = _close_at_or_before(benchmark_close, as_of_dates)
    benchmark_close_future = _close_at_or_before(benchmark_close, future_dates)

    targeted[BENCHMARK_RETURN_COLUMN] = benchmark_close_future / benchmark_close_as_of - 1
    targeted[ALPHA_TARGET_COLUMN] = (
        targeted[FUTURE_RETURN_COLUMN] - targeted[BENCHMARK_RETURN_COLUMN]
    )
    targeted[OUTPERFORM_TARGET_COLUMN] = _binary_target(
        targeted[ALPHA_TARGET_COLUMN],
        targeted[ALPHA_TARGET_COLUMN] > 0,
    )

    return targeted


def build_monthly_samples(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    _require_datetime_index(df)
    if not ticker.strip():
        raise ValueError("ticker must not be empty.")
    if FUTURE_RETURN_COLUMN not in df.columns:
        raise ValueError("Run add_targets before build_monthly_samples.")

    required_columns = ["Close", FUTURE_RETURN_COLUMN]
    for optional_column in (BENCHMARK_RETURN_COLUMN, ALPHA_TARGET_COLUMN):
        if optional_column in df.columns:
            required_columns.append(optional_column)

    complete = df.dropna(subset=required_columns).copy()
    if complete.empty:
        return complete

    if "ticker" in complete.columns:
        complete = complete.drop(columns=["ticker"])
    complete.insert(0, "ticker", ticker)

    return complete.groupby(complete.index.to_period("M"), group_keys=False).tail(1)
