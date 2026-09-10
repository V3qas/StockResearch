import numpy as np
import pandas as pd
import pytest

from src.backtest import run_cross_sectional_walk_forward_backtest, summarize_backtest
from src.datasets import build_ticker_samples, combine_ticker_samples
from src.metrics import ranking_metrics_by_date
from src.targets import add_benchmark_targets, add_targets, build_monthly_samples
from src.calendars import benchmark_coverage, reference_sessions
from src.features import FEATURE_COLUMNS


def prices_until(end):
    dates = pd.bdate_range("2017-01-02", end)
    return pd.DataFrame({"Close": 100.0 + np.arange(len(dates))}, index=dates)


def test_market_holidays_share_decision_and_target_dates():
    us = prices_until("2020-03-31")
    german = us.drop(pd.to_datetime(["2018-12-31", "2019-01-30"]))
    a = build_ticker_samples("US", us, us, 20)
    b = build_ticker_samples("DE", german, us, 20)
    decision = pd.Timestamp("2018-12-31")
    assert a.loc[decision, "price_date"] == decision
    assert b.loc[decision, "price_date"] == pd.Timestamp("2018-12-28")
    assert a.loc[decision, "future_target_date"] == b.loc[decision, "future_target_date"]
    assert b.loc[decision, "future_price_date"] < b.loc[decision, "future_target_date"]
    combined = combine_ticker_samples([a, b])
    assert combined.loc[[decision], "ticker"].nunique() == 2


def test_appended_future_data_never_moves_existing_monthly_samples():
    early = prices_until("2019-01-15")
    late = prices_until("2019-03-31")
    before = build_ticker_samples("TEST", early, early, 20)
    after = build_ticker_samples("TEST", late, late, 20)
    assert before.index.is_month_end.all()
    assert after.index.is_month_end.all()
    assert before.loc["2018-12-31", "label_status"] == "pending"
    assert after.loc["2018-12-31", "label_status"] == "observed"
    stable_columns = ["price_date", "Close", "future_target_date", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(before[stable_columns], after.loc[before.index, stable_columns])
    observed = before.loc[before.label_status == "observed"].drop(columns="snapshot_date")
    pd.testing.assert_frame_equal(observed, after.loc[observed.index].drop(columns="snapshot_date"))


def test_sampling_does_not_backfill_a_missing_feature_or_label():
    prices = prices_until("2019-02-28")
    prices["feature"] = 1.0
    prices["future_return_12m"] = 0.1
    prices.loc["2018-12-31", ["feature", "future_return_12m"]] = np.nan
    samples = build_monthly_samples(prices, "TEST")
    assert samples.loc["2018-12-31", "price_date"] == pd.Timestamp("2018-12-31")
    assert pd.isna(samples.loc["2018-12-31", "feature"])
    assert pd.isna(samples.loc["2018-12-31", "future_return_12m"])


def test_stale_stock_prices_and_unfinished_month_are_excluded():
    benchmark = prices_until("2019-02-15")
    stock = benchmark.loc[:"2018-12-10"]
    result = build_ticker_samples("TEST", stock, benchmark, 20, snapshot_date=benchmark.index.max())
    assert pd.Timestamp("2018-12-31") not in result.index
    assert result.index.is_month_end.all()
    assert result.index.max() < pd.Timestamp("2019-02-01")


@pytest.mark.parametrize("stale_start", [False, True])
def test_stale_benchmark_prices_leave_alpha_and_class_missing(stale_start):
    prices = prices_until("2019-03-31")
    targeted = add_targets(prices, forecast_days=20).loc[[pd.Timestamp("2018-12-31")]]
    dates = ["2018-12-01", "2019-01-28"] if stale_start else ["2018-12-31"]
    benchmark = pd.DataFrame({"Close": [1000.0] * len(dates)}, index=pd.to_datetime(dates))
    result = add_benchmark_targets(targeted, benchmark)
    assert pd.isna(result.alpha_12m.iloc[0])
    assert pd.isna(result.future_outperform_12m.iloc[0])


def test_benchmark_holiday_uses_bounded_prior_price_and_records_date():
    samples = pd.DataFrame({"future_return_12m": [0.2], "future_target_date": pd.to_datetime(["2020-02-29"])}, index=pd.to_datetime(["2020-01-31"]))
    benchmark = pd.DataFrame({"Close": [100.0, 110.0, 999.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-02"]))
    result = add_benchmark_targets(samples, benchmark)
    assert result.benchmark_return_12m.iloc[0] == pytest.approx(0.1)
    assert result.benchmark_future_price_date.iloc[0] == pd.Timestamp("2020-02-28")


@pytest.mark.parametrize("probabilities,count,accuracy", [
    ([np.nan, np.nan, np.nan], 0, np.nan),
    ([np.nan, 0.9, 0.1], 2, 1.0),
])
def test_classification_counts_only_available_prediction_target_pairs(probabilities, count, accuracy):
    predictions = pd.DataFrame({
        "alpha_12m": [-0.1, 0.2, -0.3], "predicted_alpha_12m": [0.0] * 3,
        "future_outperform_12m": [0, 1, 0],
        "predicted_probability_outperform_12m": probabilities,
    })
    summary = summarize_backtest(predictions)
    assert summary.classification_samples == count
    if count:
        assert summary.classification_accuracy == accuracy
    else:
        for metric in ["classification_accuracy", "precision", "recall", "brier_score", "roc_auc"]:
            assert pd.isna(summary[metric])


def test_cross_sectional_training_uses_only_mature_labels_without_duplicates():
    rows = []
    for year in range(2017, 2022):
        for month in [1, 2, 12]:
            date = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            for feature, ticker in enumerate(["A", "B", "C"]):
                rows.append({"as_of": date, "ticker": ticker, "feature": feature,
                             "alpha_12m": feature - 1.0, "future_outperform_12m": int(feature > 1),
                             "future_target_date": date + pd.DateOffset(years=1)})
    samples = pd.DataFrame(rows).set_index("as_of")
    records = []
    class Model:
        def fit(self, x, y):
            records.append(x.copy())
        def predict(self, x):
            return x.feature.to_numpy()
    result = run_cross_sectional_walk_forward_backtest(
        samples, feature_columns=["feature"], min_train_years=2, min_train_samples=3,
        return_model_factory=Model, probability_model_factory=Model,
    )
    for position, (year, fold) in enumerate(result.groupby("test_year")):
        expected = samples.loc[(samples.index.year < year) & (samples.future_target_date <= fold.index.min()), ["feature"]]
        pd.testing.assert_frame_equal(records[2 * position], expected)
        assert fold.loc[fold.ticker == "C", "predicted_rank"].eq(1).all()
        assert fold.loc[fold.ticker == "A", "predicted_rank"].eq(3).all()
    with pytest.raises(ValueError, match="future_target_date"):
        run_cross_sectional_walk_forward_backtest(samples.drop(columns="future_target_date"), feature_columns=["feature"])
    mixed = samples.copy()
    mixed.iloc[0, mixed.columns.get_loc("future_target_date")] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="same target date"):
        run_cross_sectional_walk_forward_backtest(mixed, feature_columns=["feature"])


def test_empty_combined_dataset_returns_empty_backtest():
    result = run_cross_sectional_walk_forward_backtest(combine_ticker_samples([]))
    assert result.empty
    assert "predicted_rank" in result


def test_rank_ic_is_per_date_and_undefined_for_ties_or_too_few_tickers():
    frames = []
    for date, scores in [("2020-01-31", [1, 2, 3]), ("2020-02-29", [3, 2, 1]), ("2020-03-31", [0, 0, 0]), ("2020-04-30", [1, 2])]:
        frames.append(pd.DataFrame({"ticker": list("ABC")[:len(scores)], "alpha_12m": np.arange(len(scores)), "predicted_alpha_12m": scores}, index=pd.DatetimeIndex([date] * len(scores))))
    metrics = ranking_metrics_by_date(pd.concat(frames))
    assert metrics.rank_ic.iloc[0] == pytest.approx(1.0)
    assert metrics.rank_ic.iloc[1] == pytest.approx(-1.0)
    assert pd.isna(metrics.rank_ic.iloc[2])
    assert pd.isna(metrics.rank_ic.iloc[3])
    assert list(metrics.n_tickers) == [3, 3, 3, 2]


@pytest.mark.parametrize("missing_date", ["2018-12-31", "2019-01-15", "2019-01-30"])
def test_benchmark_gap_does_not_move_horizon_or_hide_eligible_sample(missing_date):
    stock = prices_until("2020-03-31")
    benchmark = stock.drop(pd.Timestamp(missing_date))
    complete = build_ticker_samples("A", stock, stock, 20)
    gapped = build_ticker_samples("A", stock, benchmark, 20)
    decision = pd.Timestamp("2018-12-31")
    assert gapped.loc[decision, "future_target_date"] == pd.Timestamp("2019-01-30")
    pd.testing.assert_series_equal(complete.future_target_date, gapped.future_target_date)
    if missing_date == "2019-01-15":
        assert gapped.loc[decision, "alpha_12m"] == complete.loc[decision, "alpha_12m"]
    else:
        assert gapped.loc[decision, "label_status"] == "missing"
        assert pd.isna(gapped.loc[decision, "alpha_12m"])
        assert pd.isna(gapped.loc[decision, "future_outperform_12m"])
    sessions = reference_sessions(stock.index.min(), stock.index.max(), 20)
    audit = benchmark_coverage(benchmark, sessions, stock.index.min(), stock.index.max())
    assert not audit.loc[missing_date, "available"]


def test_independent_calendar_excludes_nyse_holidays_and_retains_pending_target():
    dates = pd.to_datetime(["2019-01-18"])
    prices = pd.DataFrame({"Close": [100.0]}, index=dates)
    result = add_targets(prices, forecast_days=1)
    # Weekend plus MLK Day: a close only four days old must not fabricate
    # a future outcome for Tuesday when the snapshot is still Friday.
    assert result.future_target_date.iloc[0] == pd.Timestamp("2019-01-22")
    assert pd.isna(result.future_price_12m.iloc[0])
    assert pd.isna(result.future_positive_12m.iloc[0])


def test_missing_future_stock_price_stays_in_predictions_and_ranks():
    from src.baselines import ConstantProbabilityModel, MomentumRankingModel

    full = prices_until("2020-03-31")
    unavailable = full.drop(full.loc["2019-01-18":"2019-02-10"].index)
    frames = [build_ticker_samples(ticker, prices, full, 20) for ticker, prices in (
        ("A", full), ("B", full * 2), ("C", unavailable),
    )]
    samples = combine_ticker_samples(frames)
    predictions = run_cross_sectional_walk_forward_backtest(
        samples, min_train_years=1, min_train_samples=2,
        return_model_factory=MomentumRankingModel,
        probability_model_factory=ConstantProbabilityModel,
    )
    decision = pd.Timestamp("2018-12-31")
    group = predictions.loc[[decision]]
    assert set(group.ticker) == {"A", "B", "C"}
    assert group.predicted_rank.notna().all()
    missing = group.loc[group.ticker == "C"].iloc[0]
    assert missing.label_status == "missing"
    assert pd.isna(missing.alpha_12m)
    assert missing.predicted_probability_outperform_12m == 0.5
    metrics = ranking_metrics_by_date(group, min_tickers=2).iloc[0]
    assert metrics.n_tickers == 3
    assert metrics.n_evaluated_tickers == 2
    assert metrics.n_missing_targets == 1
    assert metrics.label_coverage == pytest.approx(2 / 3)
    assert pd.isna(metrics.rank_ic)
    assert summarize_backtest(group).samples == 2


def test_unknown_training_labels_are_excluded_but_unknown_test_labels_are_predicted():
    from src.baselines import ConstantProbabilityModel

    samples = pd.DataFrame({
        "ticker": ["A"] * 5,
        "feature": [0, 1, 2, 3, 4],
        "alpha_12m": [np.nan, 0.2, 9999.0, np.nan, np.nan],
        "future_outperform_12m": pd.array([pd.NA, 1, 1, pd.NA, pd.NA], dtype="Int64"),
        "future_target_date": pd.to_datetime(["2018-01-31", "2019-01-31", "2020-12-31", "2021-01-31", "2021-02-28"]),
    }, index=pd.to_datetime(["2017-01-31", "2018-01-31", "2019-12-31", "2020-01-31", "2020-02-29"]))

    class RecordingModel:
        def fit(self, x, y):
            assert x.feature.tolist() == [1]
            assert y.tolist() == [0.2]
        def predict(self, x):
            return np.full(len(x), 0.2)

    result = run_cross_sectional_walk_forward_backtest(
        samples, feature_columns=["feature"], min_train_years=3, min_train_samples=1,
        return_model_factory=RecordingModel, probability_model_factory=ConstantProbabilityModel,
    )
    assert len(result) == 2
    assert result.alpha_12m.isna().all()
    assert result.predicted_alpha_12m.eq(0.2).all()
