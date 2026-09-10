from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    PREDICTED_PROBABILITY_COLUMN,
    PREDICTED_RETURN_COLUMN,
    run_walk_forward_backtest,
    summarize_backtest,
    yearly_walk_forward_splits,
)


class RecordingMeanRegressor:
    def __init__(self, trained_until: list[pd.Timestamp]) -> None:
        self.trained_until = trained_until
        self.mean_return = 0.0

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.trained_until.append(x_train.index.max())
        self.mean_return = float(y_train.mean())

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mean_return, len(x_test))


class MeanProbabilityClassifier:
    classes_ = np.array([0, 1])

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.positive_rate = float(y_train.mean())

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        positive = np.repeat(self.positive_rate, len(x_test))
        negative = 1 - positive
        return np.column_stack([negative, positive])


def test_yearly_walk_forward_splits_are_expanding_by_calendar_year() -> None:
    dates = pd.bdate_range("2018-01-01", "2022-12-31")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)

    splits = list(yearly_walk_forward_splits(df, min_train_years=2))

    assert [test_index[0].year for _, test_index in splits] == [2020, 2021, 2022]
    assert splits[0][0].min() == pd.Timestamp("2018-01-01")
    assert splits[0][0].max().year == 2019


def test_backtest_uses_only_labels_known_at_test_year_start() -> None:
    dates = pd.date_range("2010-01-31", "2015-12-31", freq="ME")
    values = np.arange(len(dates))
    future_return = pd.Series(((values % 4) - 1.5) / 10, index=dates)
    trained_until: list[pd.Timestamp] = []
    samples = pd.DataFrame(
        {
            "feature": values,
            "future_target_date": dates + pd.DateOffset(years=1),
            "future_return_12m": future_return,
            "future_positive_12m": (future_return > 0).astype(int),
        },
        index=dates,
    )

    predictions = run_walk_forward_backtest(
        samples,
        feature_columns=["feature"],
        min_train_years=3,
        min_train_samples=12,
        return_model_factory=lambda: RecordingMeanRegressor(trained_until),
        probability_model_factory=MeanProbabilityClassifier,
    )

    assert not predictions.empty
    assert trained_until[0] <= pd.Timestamp("2012-01-31")
    assert predictions[PREDICTED_RETURN_COLUMN].notna().all()
    assert predictions[PREDICTED_PROBABILITY_COLUMN].between(0, 1).all()


def test_summarize_backtest_returns_regression_and_classification_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "future_return_12m": [0.10, -0.05, 0.20, -0.10],
            "predicted_return_12m": [0.08, -0.01, 0.18, 0.02],
            "future_positive_12m": [1, 0, 1, 0],
            "predicted_probability_positive_12m": [0.80, 0.40, 0.70, 0.60],
        }
    )

    summary = summarize_backtest(predictions)

    assert summary["samples"] == 4
    assert summary["mae"] == pytest.approx(0.05)
    assert summary["classification_accuracy"] == pytest.approx(0.75)
    assert summary["roc_auc"] == pytest.approx(1.0)
