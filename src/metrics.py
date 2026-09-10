from __future__ import annotations

import numpy as np
import pandas as pd


def _paired_values(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    return pd.concat([y_true, y_pred], axis=1).dropna()


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    errors = _paired_values(y_true, y_pred)
    if errors.empty:
        return float("nan")
    return float(np.mean(np.abs(errors.iloc[:, 0] - errors.iloc[:, 1])))


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if values.empty:
        return float("nan")
    return float((np.sign(values.iloc[:, 0]) == np.sign(values.iloc[:, 1])).mean())


def correlation(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if len(values) < 2 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(values.iloc[:, 0].corr(values.iloc[:, 1]))


def ranking_metrics_by_date(
    predictions: pd.DataFrame,
    target_column: str = "alpha_12m",
    prediction_column: str = "predicted_alpha_12m",
    min_tickers: int = 3,
) -> pd.DataFrame:
    """Cross-sectional IC and Spearman IC, with one equal-weight observation per date.

    Constant predictions (e.g. zero alpha) have undefined IC, not an arbitrary
    ticker-order ranking. Partial outcomes are reported as incomplete, without
    computing an IC on a subset selected by future quote availability.
    """
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if min_tickers < 2:
        raise ValueError("min_tickers must be at least 2.")
    if predictions.reset_index(drop=True).assign(as_of=predictions.index).duplicated(["as_of", "ticker"]).any():
        raise ValueError("Duplicate as_of/ticker predictions.")
    rows = []
    for as_of, group in predictions.groupby(level=0, sort=True):
        valid = group.dropna(subset=["ticker", target_column, prediction_column])
        count = group.ticker.nunique()
        evaluated_count = valid.ticker.nunique()
        complete = evaluated_count == count and count >= min_tickers
        actual, predicted = valid[target_column], valid[prediction_column]
        rows.append({
            "as_of": as_of,
            "n_tickers": count,
            "n_evaluated_tickers": evaluated_count,
            "n_missing_targets": int(group[target_column].isna().sum()),
            "label_coverage": float(group[target_column].notna().mean()),
            "ic": correlation(actual, predicted) if complete else np.nan,
            "rank_ic": correlation(actual.rank(), predicted.rank()) if complete else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["n_tickers", "n_evaluated_tickers", "n_missing_targets", "label_coverage", "ic", "rank_ic"], index=pd.DatetimeIndex([], name="as_of"))
    return pd.DataFrame(rows).set_index("as_of")


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if len(values) < 2:
        return float("nan")

    actual = values.iloc[:, 0].astype(float)
    predicted = values.iloc[:, 1].astype(float)
    total_sum_squares = float(((actual - actual.mean()) ** 2).sum())
    if total_sum_squares == 0:
        return float("nan")

    residual_sum_squares = float(((actual - predicted) ** 2).sum())
    return 1 - residual_sum_squares / total_sum_squares


def binary_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if values.empty:
        return float("nan")
    return float((values.iloc[:, 0].astype(int) == values.iloc[:, 1].astype(int)).mean())


def precision_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if values.empty:
        return float("nan")

    actual = values.iloc[:, 0].astype(int)
    predicted = values.iloc[:, 1].astype(int)
    true_positives = int(((actual == 1) & (predicted == 1)).sum())
    false_positives = int(((actual == 0) & (predicted == 1)).sum())
    denominator = true_positives + false_positives
    if denominator == 0:
        return float("nan")
    return true_positives / denominator


def recall_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = _paired_values(y_true, y_pred)
    if values.empty:
        return float("nan")

    actual = values.iloc[:, 0].astype(int)
    predicted = values.iloc[:, 1].astype(int)
    true_positives = int(((actual == 1) & (predicted == 1)).sum())
    false_negatives = int(((actual == 1) & (predicted == 0)).sum())
    denominator = true_positives + false_negatives
    if denominator == 0:
        return float("nan")
    return true_positives / denominator


def brier_score(y_true: pd.Series, probabilities: pd.Series) -> float:
    values = _paired_values(y_true, probabilities)
    if values.empty:
        return float("nan")

    actual = values.iloc[:, 0].astype(float)
    predicted_probability = values.iloc[:, 1].astype(float)
    return float(((predicted_probability - actual) ** 2).mean())


def roc_auc_score(y_true: pd.Series, probabilities: pd.Series) -> float:
    values = _paired_values(y_true, probabilities)
    if values.empty:
        return float("nan")

    actual = values.iloc[:, 0].astype(int)
    scores = values.iloc[:, 1].astype(float)
    positives = int((actual == 1).sum())
    negatives = int((actual == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = scores.rank(method="average")
    positive_rank_sum = float(ranks[actual == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def calibration_table(
    probabilities: pd.Series,
    outcomes: pd.Series,
    bin_width: float = 0.10,
) -> pd.DataFrame:
    if not 0 < bin_width <= 1:
        raise ValueError("bin_width must be in (0, 1].")

    values = pd.DataFrame(
        {
            "probability": probabilities,
            "outcome": outcomes,
        }
    ).dropna()
    if values.empty:
        return pd.DataFrame(
            columns=[
                "bin",
                "count",
                "mean_predicted_probability",
                "actual_rate",
                "calibration_error",
            ]
        )

    bins = np.arange(0, 1 + bin_width, bin_width)
    values["bin"] = pd.cut(
        values["probability"],
        bins=bins,
        include_lowest=True,
        right=True,
    )

    grouped = values.groupby("bin", observed=True)
    table = grouped.agg(
        count=("outcome", "size"),
        mean_predicted_probability=("probability", "mean"),
        actual_rate=("outcome", "mean"),
    ).reset_index()
    table["calibration_error"] = (
        table["mean_predicted_probability"] - table["actual_rate"]
    ).abs()

    return table
