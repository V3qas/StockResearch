import json

import numpy as np
import pandas as pd
import pytest

from src.signal_report import (
    decile_returns_by_date, paired_rank_comparison, price_jump_audit,
    rank_deciles, summarize_deciles,
)
from src.universe import load_universe


def predictions(count=100, date="2020-01-31", target=0.1):
    return pd.DataFrame({
        "model": "xgboost", "ticker": [f"T{i:03}" for i in range(count)],
        "ranking_score": np.arange(count, dtype=float), "alpha_12m": target,
    }, index=pd.DatetimeIndex([date] * count, name="as_of"))


def test_deciles_follow_score_direction_and_keep_ties_independent_of_row_order():
    frame = predictions()
    ranked = rank_deciles(frame)
    assert ranked.groupby("decile").size().eq(10).all()
    assert ranked.loc[ranked.ticker == "T099", "decile"].iloc[0] == 1
    assert ranked.loc[ranked.ticker == "T000", "decile"].iloc[0] == 10
    frame.iloc[85:95, frame.columns.get_loc("ranking_score")] = 90
    tied = rank_deciles(frame).set_index("ticker").decile
    assert tied.iloc[85:95].nunique() == 1
    shuffled = rank_deciles(frame.sample(frac=1, random_state=42)).set_index("ticker").decile
    pd.testing.assert_series_equal(tied.sort_index(), shuffled.sort_index())


def test_unknown_outcomes_do_not_change_buckets_and_suppress_entire_date_returns():
    frame = predictions()
    before = rank_deciles(frame)
    frame.iloc[99, frame.columns.get_loc("alpha_12m")] = np.nan
    after = rank_deciles(frame)
    pd.testing.assert_series_equal(before.decile, after.decile)
    dated = decile_returns_by_date(after)
    assert dated.n_tickers.sum() == 100
    assert dated.n_observed.sum() == 99
    assert dated.mean_excess_return.isna().all()
    assert not dated.complete_date.any()


def test_decile_summary_weights_dates_equally_despite_different_universe_sizes():
    frame = pd.concat([predictions(100, target=0.1), predictions(20, "2020-02-29", 0.5)])
    summary = summarize_deciles(decile_returns_by_date(rank_deciles(frame)))
    assert summary.mean_excess_return.to_numpy() == pytest.approx([0.3] * 10)
    assert summary.evaluated_dates.eq(2).all()


@pytest.mark.parametrize("condition", ["constant", "small", "missing_score", "infinite_score"])
def test_unrankable_dates_have_no_arbitrary_buckets(condition):
    frame = predictions(9 if condition == "small" else 100)
    if condition == "constant":
        frame["ranking_score"] = 0.0
    elif condition in ("missing_score", "infinite_score"):
        frame.iloc[0, frame.columns.get_loc("ranking_score")] = np.nan if condition == "missing_score" else np.inf
    assert rank_deciles(frame).decile.isna().all()


def test_empty_buckets_from_ties_cannot_create_curve_using_different_dates():
    frame = predictions()
    frame["ranking_score"] = np.repeat([0.0, 1.0], 50)
    dated = decile_returns_by_date(rank_deciles(frame))
    assert dated.n_tickers.sum() == 100
    assert dated.mean_excess_return.isna().all()


def test_pairwise_ic_uses_only_shared_defined_months():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2021-01-31"])
    ranking = pd.concat([
        pd.DataFrame({"model": "xgboost", "rank_ic": [0.1, 0.9, -0.1]}, index=dates),
        pd.DataFrame({"model": "momentum_120d", "rank_ic": [0.2, np.nan, -0.2]}, index=dates),
        pd.DataFrame({"model": "zero_alpha", "rank_ic": [np.nan] * 3}, index=dates),
    ])
    ranking.index.name = "as_of"
    result = paired_rank_comparison(ranking)
    row = result.loc["momentum_120d"]
    assert row.paired_dates == 2
    assert row.mean_difference == pytest.approx(0.0)
    assert row.positive_difference_fraction == 0.5
    assert row.years_ahead == 1
    assert row.paired_years == 2
    assert result.loc["zero_alpha", "paired_dates"] == 0
    assert pd.isna(result.loc["zero_alpha", "mean_difference"])


def test_price_audit_flags_both_directions_without_mutation():
    prices = pd.DataFrame({"Close": [100.0, 200.0, 50.0, 55.0]}, index=pd.date_range("2020-01-01", periods=4))
    original = prices.copy()
    result = price_jump_audit(prices, "TEST")
    assert result.daily_return.tolist() == [1.0, -0.75]
    pd.testing.assert_frame_equal(prices, original)


def test_frozen_us_universe_has_300_unique_symbols_and_provenance():
    from main import validate_tickers
    from src.config import PROJECT_ROOT

    universe = load_universe(PROJECT_ROOT / "universes/us_large_cap_300.json")
    assert len(validate_tickers(universe["tickers"])) == 300
    assert universe["membership"] == "fixed_current_snapshot"
    assert len(universe["source_sha256"]) == 64
    assert universe["limitations"]


@pytest.mark.parametrize("payload", [[], {}, {"tickers": []}, {"tickers": "AAPL"}, {"tickers": [42]}])
def test_invalid_universe_is_rejected(tmp_path, payload):
    path = tmp_path / "universe.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Universe"):
        load_universe(path)


def test_replay_rejects_modified_frozen_inputs(tmp_path):
    from replay_signal import verify_inputs
    from src.experiments import ExperimentRun

    with ExperimentRun(tmp_path / "runs", {}) as run:
        for path in [run.raw / "benchmark.parquet", run.raw / "reference_calendar.parquet", run.processed / "all_samples.parquet"]:
            pd.DataFrame({"value": [1.0]}).to_parquet(path)
    assert verify_inputs(run.path)["run_id"] == run.run_id
    pd.DataFrame({"value": [2.0]}).to_parquet(run.raw / "benchmark.parquet")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_inputs(run.path)


def test_all_ticker_cannot_collide_with_combined_dataset_on_windows(tmp_path):
    from src.datasets import save_combined_samples, save_ticker_samples

    stock = pd.DataFrame({"ticker": ["ALL"], "value": [1.0]})
    combined = pd.DataFrame({"ticker": ["ALL", "OTHER"], "value": [1.0, 2.0]})
    stock_path = save_ticker_samples(stock, "ALL", tmp_path)
    combined_path = save_combined_samples(combined, tmp_path)
    assert stock_path.name.casefold() != combined_path.name.casefold()
    pd.testing.assert_frame_equal(pd.read_parquet(stock_path), stock)
    pd.testing.assert_frame_equal(pd.read_parquet(combined_path), combined)
