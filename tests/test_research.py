import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    ConstantReturnModel, ConstantProbabilityModel, HistoricalMeanModel,
    HistoricalFrequencyModel, MomentumRankingModel, create_ridge_model,
    create_logistic_model,
)
from src.datasets import combine_ticker_samples
from src.research import ModelSpec, run_model_comparison


def sample_frame():
    rows = []
    for date in pd.date_range("2017-01-31", "2022-12-31", freq="ME"):
        for ticker, feature in [("A", -0.1), ("B", 0.0), ("C", 0.1)]:
            rows.append({"as_of": date, "ticker": ticker, "return_120d": feature,
                         "alpha_12m": feature + 0.01 * (date.year - 2017),
                         "future_outperform_12m": int(feature >= 0),
                         "future_target_date": date + pd.DateOffset(years=1)})
    return pd.DataFrame(rows).set_index("as_of")


@pytest.mark.parametrize(
    "tickers, message",
    [
        (["TSLA", "tsla"], "case-insensitive"),
        (["BRK.B", "BRK/B"], "distinct cache filenames"),
    ],
)
def test_ticker_validation_rejects_cross_platform_filename_collisions(tickers, message):
    import main

    with pytest.raises(ValueError, match=message):
        main.validate_tickers(tickers)


def test_ticker_validation_strips_surrounding_whitespace():
    import main

    assert main.validate_tickers([" TSLA ", "BABA"]) == ["TSLA", "BABA"]


def test_models_share_folds_and_baselines_use_training_history_only():
    samples = sample_frame()
    specs = [
        ModelSpec("zero", ConstantReturnModel, ConstantProbabilityModel),
        ModelSpec("mean", HistoricalMeanModel, HistoricalFrequencyModel),
        ModelSpec("ridge", create_ridge_model, create_logistic_model),
        ModelSpec("momentum", MomentumRankingModel, lambda: ConstantProbabilityModel(np.nan), True),
    ]
    comparison = run_model_comparison(
        samples, min_train_years=2, min_train_samples=3,
        feature_columns=["return_120d"], model_specs=specs,
    )
    predictions = comparison.predictions
    first_keys = None
    for _, group in predictions.groupby("model"):
        keys = pd.MultiIndex.from_arrays([group.index, group.ticker])
        if first_keys is not None:
            assert keys.equals(first_keys)
        first_keys = keys
    for year, fold in predictions.loc[predictions.model == "mean"].groupby("test_year"):
        known = samples.loc[(samples.index.year < year) & (samples.future_target_date <= fold.index.min())]
        assert fold.predicted_alpha_12m.eq(known.alpha_12m.mean()).all()
        assert fold.predicted_probability_outperform_12m.eq(known.future_outperform_12m.mean()).all()
    momentum = predictions.loc[predictions.model == "momentum"]
    assert momentum.predicted_alpha_12m.isna().all()
    assert momentum.ranking_score.notna().all()
    assert pd.isna(comparison.summary.loc["momentum", "mae"])
    assert comparison.summary.loc["momentum", "mean_rank_ic"] == pytest.approx(1.0)
    assert comparison.summary.loc["momentum", "median_rank_ic"] == pytest.approx(1.0)
    assert comparison.summary.loc["momentum", "positive_rank_ic_fraction"] == 1.0
    assert pd.isna(comparison.summary.loc["zero", "positive_rank_ic_fraction"])
    assert comparison.summary.loc["zero", "ic_dates"] == 0
    assert comparison.summary.loc["zero", "brier_score"] == pytest.approx(0.25)
    assert set(comparison.yearly_summary.index.get_level_values("test_year")) == {2019, 2020, 2021, 2022}


def test_comparison_of_empty_dataset_has_zero_counts_and_undefined_metrics():
    comparison = run_model_comparison(combine_ticker_samples([]))
    assert comparison.predictions.empty
    assert comparison.summary.samples.eq(0).all()
    assert comparison.summary.mae.isna().all()
    assert comparison.summary.ic_dates.eq(0).all()
    assert comparison.classification_comparison.classification_samples.eq(0).all()


@pytest.mark.parametrize("min_train_years", [2, 50])
def test_cli_writes_comparable_artifacts_for_one_explicit_snapshot(monkeypatch, tmp_path, min_train_years):
    from argparse import Namespace
    import json
    import main

    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    calls = []
    def fake_get_stock(ticker, start_date, end_date, force_download):
        calls.append((ticker, end_date))
        dates = pd.bdate_range("2010-01-01", "2017-12-31")
        frame = pd.DataFrame({"Close": 100.0 + np.arange(len(dates)) * (1 + len(ticker))}, index=dates)
        return frame
    monkeypatch.setattr(main, "get_stock", fake_get_stock)
    monkeypatch.setattr(main, "default_model_specs", lambda: [ModelSpec("xgboost", ConstantReturnModel, ConstantProbabilityModel)])
    monkeypatch.setattr(main, "parse_args", lambda: Namespace(
        tickers=["A", "BB", "CCC"], start_date="2010-01-01", end_date="2018-01-01",
        force_download=False, samples=0, backtest=False, cross_sectional_backtest=False,
        compare_models=True, min_train_years=min_train_years, min_train_samples=3,
        max_price_age_days=7, min_ranking_tickers=3,
    ))
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()
    old_output = output_dir / "cross_sectional_backtest_predictions.parquet"
    pd.DataFrame({"stale": [1]}).to_parquet(old_output)
    main.main()
    assert {boundary for _, boundary in calls} == {"2018-01-01"}
    pointer = json.loads((main.RUNS_DIR / "latest.json").read_text())
    run_dir = main.RUNS_DIR / pointer["run_id"]
    output_dir = run_dir / "predictions"
    saved = pd.read_parquet(run_dir / "processed/all_samples.parquet")
    assert saved.index.is_month_end.all()
    assert saved.groupby(level=0).future_target_date.nunique().eq(1).all()
    assert saved.future_target_date.max() > pd.Timestamp("2018-01-01")
    assert saved.loc[saved.label_status == "pending", "alpha_12m"].isna().all()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["parameters"]["end_date_exclusive"] == "2018-01-01"
    assert manifest["parameters"]["benchmark"] == "SPY"
    assert manifest["parameters"]["reference_calendar"] == "XNYS"
    assert manifest["parameters"]["cross_sectional_backtest_requested"] is False
    assert manifest["parameters"]["model_comparison_requested"] is True
    assert pd.read_parquet(old_output).stale.eq(1).all()
    assert pd.read_parquet(output_dir / old_output.name).empty == (min_train_years == 50)
    assert (run_dir / "raw/benchmark.parquet").exists()
    assert (run_dir / "processed/benchmark_coverage.csv").exists()
    assert (output_dir / "classification_common_cohort.csv").exists()
    report = (output_dir / "signal_report.md").read_text(encoding="utf-8")
    assert "Does StockResearch find signal?" in report
    assert "No complete, populated decile dates" in report
    assert "predictions/signal_report.md" in manifest["artifacts_sha256"]
    assert "model" in pd.read_csv(output_dir / "model_comparison_by_year.csv").columns
    for name in ["model_comparison_summary.csv", "model_comparison_by_year.csv", "ranking_metrics_by_date.parquet"]:
        assert (output_dir / name).exists()


@pytest.mark.parametrize("training_class", [0, 1, None])
def test_single_class_baselines_keep_valid_probabilities(training_class):
    from src.backtest import run_cross_sectional_walk_forward_backtest

    samples = sample_frame()
    training = samples.index.year < 2019
    samples["future_outperform_12m"] = samples.future_outperform_12m.astype("Int64")
    samples.loc[training, "future_outperform_12m"] = training_class if training_class is not None else pd.NA
    for model, expected in [(ConstantProbabilityModel, 0.5), (HistoricalFrequencyModel, training_class)]:
        result = run_cross_sectional_walk_forward_backtest(
            samples, feature_columns=["return_120d"], min_train_years=2, min_train_samples=3,
            return_model_factory=ConstantReturnModel, probability_model_factory=model,
        )
        first = result.loc[result.test_year == 2019, "predicted_probability_outperform_12m"]
        assert len(first) == 36
        assert first.isna().all() if expected is None else first.eq(expected).all()


def test_classification_comparison_uses_common_cohort_without_suppressing_baselines():
    samples = sample_frame()
    samples.loc[samples.index.year < 2019, "future_outperform_12m"] = 1

    class NeedsTwoClasses(HistoricalFrequencyModel):
        supports_single_class = False
        supports_empty_training = False
        def fit(self, x, y):
            assert y.nunique() == 2
            super().fit(x, y)

    result = run_model_comparison(
        samples, feature_columns=["return_120d"], min_train_years=2, min_train_samples=3,
        model_specs=[
            ModelSpec("baseline", ConstantReturnModel, ConstantProbabilityModel),
            ModelSpec("classifier", ConstantReturnModel, NeedsTwoClasses),
            ModelSpec("momentum", MomentumRankingModel, ConstantProbabilityModel, True),
        ],
    )
    assert result.summary.loc["baseline", "classification_samples"] == 144
    assert result.summary.loc["classifier", "classification_samples"] == 108
    common = result.classification_comparison
    assert set(common.index) == {"baseline", "classifier"}
    assert common.classification_samples.eq(108).all()
    assert common.loc["baseline", "brier_score"] == pytest.approx(0.25)
    predictions = result.predictions
    observed = predictions.loc[(predictions.model == "classifier") & predictions.predicted_probability_outperform_12m.notna()]
    expected = ((observed.future_outperform_12m - observed.predicted_probability_outperform_12m) ** 2).mean()
    assert common.loc["classifier", "brier_score"] == pytest.approx(expected)
