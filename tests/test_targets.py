from __future__ import annotations

import pandas as pd
import pytest

from src.targets import add_targets, build_monthly_samples


def test_add_targets_uses_exact_forecast_day_shift() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    df = pd.DataFrame({"Close": range(100, 360)}, index=dates)

    result = add_targets(df, forecast_days=252)

    expected_return = df["Close"].iloc[252] / df["Close"].iloc[0] - 1
    assert result["future_return_12m"].iloc[0] == pytest.approx(expected_return)
    assert result["future_target_date"].iloc[0] == dates[252]
    assert pd.isna(result["future_return_12m"].iloc[-1])
    assert pd.isna(result["future_positive_12m"].iloc[-1])


def test_build_monthly_samples_keeps_last_trading_day_per_month() -> None:
    dates = pd.bdate_range("2020-01-01", periods=420)
    df = pd.DataFrame({"Close": range(100, 520)}, index=dates)
    targeted = add_targets(df, forecast_days=252)

    samples = build_monthly_samples(targeted)

    assert samples.index.is_monotonic_increasing
    assert samples.index[0] == pd.Timestamp("2020-01-31")
    assert samples.index.to_period("M").is_unique
