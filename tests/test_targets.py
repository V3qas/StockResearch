from __future__ import annotations

import pandas as pd
import pytest

from src.targets import add_benchmark_targets, add_targets, build_monthly_samples
from src.calendars import reference_sessions


def nyse_dates():
    sessions = reference_sessions(pd.Timestamp("2020-01-02"), pd.Timestamp("2021-12-31"))
    return sessions[sessions >= "2020-01-02"][:260]


def test_add_targets_uses_exact_forecast_day_shift() -> None:
    dates = nyse_dates()
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

    samples = build_monthly_samples(targeted, ticker="TEST")

    assert samples.index.is_monotonic_increasing
    assert samples.index[0] == pd.Timestamp("2020-01-31")
    assert samples.index.to_period("M").is_unique
    assert samples["ticker"].eq("TEST").all()


def test_add_benchmark_targets_creates_12m_alpha() -> None:
    dates = nyse_dates()
    df = pd.DataFrame({"Close": range(100, 360)}, index=dates)
    targeted = add_targets(df, forecast_days=252)
    benchmark = pd.DataFrame(
        {"Close": [1_000.0, 1_100.0]},
        index=pd.to_datetime([dates[0], dates[252]]),
    )

    result = add_benchmark_targets(targeted, benchmark)

    assert result["benchmark_return_12m"].iloc[0] == pytest.approx(0.10)
    assert result["alpha_12m"].iloc[0] == pytest.approx(
        result["future_return_12m"].iloc[0] - 0.10
    )
    assert result["future_outperform_12m"].iloc[0] == 1
    assert pd.isna(result["future_outperform_12m"].iloc[-1])
