from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import (
    ModelFactory, PREDICTED_RETURN_COLUMN, PREDICTED_PROBABILITY_COLUMN,
    run_cross_sectional_walk_forward_backtest, summarize_backtest,
)
from src.baselines import (
    ConstantReturnModel, ConstantProbabilityModel, HistoricalMeanModel,
    HistoricalFrequencyModel, MomentumRankingModel, create_ridge_model,
    create_logistic_model,
)
from src.features import FEATURE_COLUMNS
from src.metrics import ranking_metrics_by_date
from src.targets import ALPHA_TARGET_COLUMN, OUTPERFORM_TARGET_COLUMN
from src.train import create_return_model, create_probability_model


@dataclass(frozen=True)
class ModelSpec:
    name: str
    return_factory: ModelFactory
    probability_factory: ModelFactory
    ranking_only: bool = False


@dataclass
class ComparisonResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    ranking_by_date: pd.DataFrame
    classification_comparison: pd.DataFrame


def default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec("xgboost", create_return_model, create_probability_model),
        ModelSpec("zero_alpha", ConstantReturnModel, ConstantProbabilityModel),
        ModelSpec("historical_mean", HistoricalMeanModel, HistoricalFrequencyModel),
        ModelSpec("ridge_logistic", create_ridge_model, create_logistic_model),
        ModelSpec("momentum_120d", MomentumRankingModel,
                  lambda: ConstantProbabilityModel(np.nan), ranking_only=True),
    ]


def _summary(predictions: pd.DataFrame, ranking: pd.DataFrame) -> pd.Series:
    result = summarize_backtest(predictions)
    result["ranking_samples"] = int(predictions.ranking_score.notna().sum())
    result["eligible_samples"] = len(predictions)
    result["observed_labels"] = int(predictions[ALPHA_TARGET_COLUMN].notna().sum())
    result["label_coverage"] = predictions[ALPHA_TARGET_COLUMN].notna().mean()
    if "label_status" in predictions:
        result["missing_labels"] = int(predictions.label_status.eq("missing").sum())
        result["pending_labels"] = int(predictions.label_status.eq("pending").sum())
    result["ranking_dates"] = len(ranking)
    result["ic_dates"] = int(ranking.rank_ic.notna().sum())
    result["mean_tickers_per_date"] = ranking.n_tickers.mean()
    result["mean_ic"] = ranking.ic.mean()
    result["mean_rank_ic"] = ranking.rank_ic.mean()
    return result


def run_model_comparison(
    samples: pd.DataFrame,
    *,
    min_train_years: int = 5,
    min_train_samples: int = 24,
    min_ranking_tickers: int = 3,
    model_specs: Sequence[ModelSpec] | None = None,
    feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS),
) -> ComparisonResult:
    """Evaluate every model on the same dated folds; never tune on test labels."""
    specs = default_model_specs() if model_specs is None else list(model_specs)
    if not specs or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Provide at least one model with unique model names.")
    predictions, summaries, yearly, rankings = [], [], [], []
    expected_keys = None
    for spec in specs:
        result = run_cross_sectional_walk_forward_backtest(
            samples, feature_columns=feature_columns,
            min_train_years=min_train_years, min_train_samples=min_train_samples,
            return_model_factory=spec.return_factory,
            probability_model_factory=spec.probability_factory,
        )
        keys = pd.MultiIndex.from_arrays([result.index, result.ticker])
        if expected_keys is not None and not keys.equals(expected_keys):
            raise ValueError("Models must be compared on identical as_of/ticker rows.")
        expected_keys = keys
        result["model"] = spec.name
        result["ranking_score"] = result[PREDICTED_RETURN_COLUMN]
        if spec.ranking_only:
            # Momentum has no calibrated alpha or probability interpretation.
            result[PREDICTED_RETURN_COLUMN] = np.nan
            result[PREDICTED_PROBABILITY_COLUMN] = np.nan
        ranking = ranking_metrics_by_date(
            result, prediction_column="ranking_score", min_tickers=min_ranking_tickers,
        )
        summary = _summary(result, ranking)
        summary["model"] = spec.name
        summaries.append(summary)
        for year, fold in result.groupby("test_year", sort=True):
            annual = _summary(fold, ranking.loc[ranking.index.year == year])
            annual["model"] = spec.name
            annual["test_year"] = int(year)
            yearly.append(annual)
        ranking["model"] = spec.name
        predictions.append(result)
        rankings.append(ranking)
    summary_frame = pd.DataFrame(summaries).set_index("model")
    annual_frame = pd.DataFrame(
        yearly, columns=["model", "test_year", *summary_frame.columns],
    ).set_index(["model", "test_year"])
    # Keep each model's own coverage above, and compare classification quality
    # on the intersection of evaluable rows. A skipped classifier fold must
    # not change the population against which another model is judged.
    probability_results = [result for spec, result in zip(specs, predictions) if not spec.ranking_only]
    common = np.ones(len(expected_keys), dtype=bool)
    for result in probability_results:
        common &= (result[OUTPERFORM_TARGET_COLUMN].notna() & result[PREDICTED_PROBABILITY_COLUMN].notna()).to_numpy()
    common_rows = []
    for spec, result in zip(specs, predictions):
        if spec.ranking_only:
            continue
        metrics = summarize_backtest(result.iloc[np.flatnonzero(common)])
        common_rows.append({
            "model": spec.name,
            "eligible_samples": len(result),
            "own_classification_samples": int(summary_frame.loc[spec.name, "classification_samples"]),
            **{key: value for key, value in metrics.items() if key in (
                "classification_samples", "classification_accuracy", "precision", "recall", "brier_score", "roc_auc",
            )},
        })
    classification = pd.DataFrame(common_rows, columns=[
        "model", "eligible_samples", "own_classification_samples", "classification_samples",
        "classification_accuracy", "precision", "recall", "brier_score", "roc_auc",
    ]).set_index("model")
    return ComparisonResult(
        predictions=pd.concat(predictions),
        summary=summary_frame,
        yearly_summary=annual_frame,
        ranking_by_date=pd.concat(rankings),
        classification_comparison=classification,
    )
