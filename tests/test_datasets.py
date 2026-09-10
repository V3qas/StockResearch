from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import RANK_COLUMN, run_cross_sectional_walk_forward_backtest
from src.datasets import build_ticker_samples, combine_ticker_samples


def test_build_ticker_samples_creates_alpha_samples_with_ticker() -> None:
    dates = pd.bdate_range("2020-01-01", periods=280)
    prices = pd.DataFrame({"Close": range(100, 380)}, index=dates)
    benchmark = pd.DataFrame({"Close": range(1_000, 1_280)}, index=dates)

    samples = build_ticker_samples(
        ticker="TEST",
        prices=prices,
        benchmark_prices=benchmark,
        forecast_days=20,
    )

    assert not samples.empty
    assert samples.index.name == "as_of"
    assert samples["ticker"].eq("TEST").all()
    assert "alpha_12m" in samples.columns
    assert "future_outperform_12m" in samples.columns


def test_combine_ticker_samples_sorts_by_date_then_ticker() -> None:
    tsla = pd.DataFrame(
        {
            "ticker": ["TSLA", "TSLA"],
            "future_return_12m": [0.10, 0.20],
            "alpha_12m": [0.01, 0.02],
            "future_outperform_12m": [1, 1],
        },
        index=pd.to_datetime(["2020-02-29", "2020-01-31"]),
    )
    baba = pd.DataFrame(
        {
            "ticker": ["BABA", "BABA"],
            "future_return_12m": [0.30, 0.40],
            "alpha_12m": [0.03, 0.04],
            "future_outperform_12m": [1, 1],
        },
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )

    combined = combine_ticker_samples([tsla, baba])

    assert list(combined.index) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-02-29"),
    ]
    assert list(combined["ticker"]) == ["BABA", "TSLA", "BABA", "TSLA"]


def test_combine_ticker_samples_rejects_duplicate_as_of_ticker_rows() -> None:
    samples = pd.DataFrame(
        {
            "ticker": ["TSLA", "TSLA"],
            "future_return_12m": [0.10, 0.20],
            "alpha_12m": [0.01, 0.02],
            "future_outperform_12m": [1, 1],
        },
        index=pd.to_datetime(["2020-01-31", "2020-01-31"]),
    )

    with pytest.raises(ValueError, match="Duplicate sample rows"):
        combine_ticker_samples([samples])


class MeanRegressor:
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.mean = float(y_train.mean())

    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        return pd.Series(self.mean, index=x_test.index)


def test_cross_sectional_backtest_predicts_and_ranks_each_test_year() -> None:
    dates = pd.to_datetime(
        [
            "2018-12-31",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
        ]
    )
    rows = []
    for date in dates:
        for ticker, feature, target in (
            ("A", 1.0, 0.10),
            ("B", 2.0, -0.10),
        ):
            rows.append(
                {
                    "as_of": date,
                    "ticker": ticker,
                    "feature": feature,
                    "alpha_12m": target,
                    "future_outperform_12m": int(target > 0),
                    "future_target_date": date + pd.DateOffset(years=1),
                }
            )
    samples = pd.DataFrame(rows).set_index("as_of")

    predictions = run_cross_sectional_walk_forward_backtest(
        samples,
        feature_columns=["feature"],
        min_train_years=2,
        min_train_samples=4,
        return_model_factory=MeanRegressor,
        probability_model_factory=MeanRegressor,
    )

    assert len(predictions) == 4
    # Constant forecasts have tied ranks; ticker order must not invent a signal.
    assert predictions[RANK_COLUMN].eq(1.5).all()
