from __future__ import annotations

import pandas as pd


TARGET_COLUMNS = [
    "future_return_12m",
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
    targeted["future_target_date"] = future_date
    targeted["future_return_12m"] = future_price / close - 1

    future_return = targeted["future_return_12m"]
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


def build_monthly_samples(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if "future_return_12m" not in df.columns:
        raise ValueError("Run add_targets before build_monthly_samples.")

    complete = df.dropna(subset=["Close", "future_return_12m"]).copy()
    if complete.empty:
        return complete

    return complete.groupby(complete.index.to_period("M"), group_keys=False).tail(1)
