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
