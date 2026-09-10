from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS
from src.metrics import (
    binary_accuracy,
    brier_score,
    correlation,
    directional_accuracy,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from src.targets import ALPHA_TARGET_COLUMN, OUTPERFORM_TARGET_COLUMN, TARGET_DATE_COLUMN


RETURN_TARGET_COLUMN = ALPHA_TARGET_COLUMN
PROBABILITY_TARGET_COLUMN = OUTPERFORM_TARGET_COLUMN
PREDICTED_RETURN_COLUMN = "predicted_alpha_12m"
PREDICTED_PROBABILITY_COLUMN = "predicted_probability_outperform_12m"
TEST_YEAR_COLUMN = "test_year"
RANK_COLUMN = "predicted_rank"

ModelFactory = Callable[[], object]


def yearly_walk_forward_splits(
    df: pd.DataFrame,
    min_train_years: int = 5,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if min_train_years < 1:
        raise ValueError("min_train_years must be at least 1.")

    years = sorted(df.index.year.unique())
    if len(years) <= min_train_years:
        return

    first_test_position = min_train_years
    for test_year in years[first_test_position:]:
        train_index = df.index[df.index.year < test_year]
        test_index = df.index[df.index.year == test_year]
        if len(train_index) and len(test_index):
            yield train_index, test_index


def _default_return_model_factory() -> object:
    from src.train import create_return_model

    return create_return_model()


def _default_probability_model_factory() -> object:
    from src.train import create_probability_model

    return create_probability_model()


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def _positive_class_probability(model: object, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 1:
            return probabilities.astype(float)

        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probabilities[:, classes.index(1)].astype(float)
        if probabilities.shape[1] == 2:
            return probabilities[:, 1].astype(float)

    predictions = model.predict(x_test)
    return np.asarray(predictions, dtype=float)


def run_walk_forward_backtest(
    samples: pd.DataFrame,
    feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS),
    target_column: str = RETURN_TARGET_COLUMN,
    probability_target_column: str = PROBABILITY_TARGET_COLUMN,
    target_date_column: str = TARGET_DATE_COLUMN,
    min_train_years: int = 5,
    min_train_samples: int = 24,
    return_model_factory: ModelFactory | None = None,
    probability_model_factory: ModelFactory | None = None,
) -> pd.DataFrame:
    """Use the same label-maturity and fitting logic for a single ticker."""
    frame = samples.copy()
    has_ticker = "ticker" in frame.columns
    if has_ticker and frame.ticker.nunique() > 1:
        raise ValueError("Use the cross-sectional backtest for multiple tickers.")
    if not has_ticker:
        frame["ticker"] = "__single__"
    result = run_cross_sectional_walk_forward_backtest(
        frame, feature_columns, target_column, probability_target_column,
        target_date_column, min_train_years, min_train_samples,
        return_model_factory, probability_model_factory,
    )
    return result.drop(columns=[RANK_COLUMN] + ([] if has_ticker else ["ticker"]))


def run_cross_sectional_walk_forward_backtest(
    samples: pd.DataFrame,
    feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS),
    target_column: str = RETURN_TARGET_COLUMN,
    probability_target_column: str = PROBABILITY_TARGET_COLUMN,
    target_date_column: str = TARGET_DATE_COLUMN,
    min_train_years: int = 5,
    min_train_samples: int = 24,
    return_model_factory: ModelFactory | None = None,
    probability_model_factory: ModelFactory | None = None,
) -> pd.DataFrame:
    """Backtest one global model and rank all tickers at each test date."""
    if not isinstance(samples.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if min_train_samples < 1:
        raise ValueError("min_train_samples must be at least 1.")

    required_columns = [*feature_columns, target_column, probability_target_column, target_date_column]
    _require_columns(samples, required_columns)
    if "ticker" not in samples.columns:
        raise ValueError("Missing columns: ['ticker']")

    if samples.reset_index(drop=True).assign(as_of=samples.index).duplicated(["as_of", "ticker"]).any():
        raise ValueError("Duplicate sample rows for the same as_of/ticker pair.")
    # Eligibility is determined at as_of. Unknown future outcomes must not
    # remove a security from the historical prediction universe.
    frame = samples.sort_index(kind="stable").dropna(subset=[*feature_columns, "ticker"]).copy()
    frame[target_date_column] = pd.to_datetime(frame[target_date_column], errors="coerce")
    if frame[target_date_column].isna().any():
        raise ValueError("Every eligible sample needs a planned target date.")
    if (frame[target_date_column] <= frame.index).any():
        raise ValueError("Target dates must be later than as_of dates.")
    if frame.groupby(level=0)[target_date_column].nunique().gt(1).any():
        raise ValueError("All tickers at an as_of must share the same target date.")
    if frame.empty:
        return frame.assign(
            **{
                TEST_YEAR_COLUMN: pd.Series(dtype="int64"),
                PREDICTED_RETURN_COLUMN: pd.Series(dtype="float64"),
                PREDICTED_PROBABILITY_COLUMN: pd.Series(dtype="float64"),
                RANK_COLUMN: pd.Series(dtype="float64"),
            }
        )

    if return_model_factory is None:
        return_model_factory = _default_return_model_factory
    if probability_model_factory is None:
        probability_model_factory = _default_probability_model_factory

    predictions = []
    for _, test_index in yearly_walk_forward_splits(
        frame,
        min_train_years=min_train_years,
    ):
        test_start = pd.Timestamp(test_index.min())
        test_year = test_start.year
        train_mask = frame.index.year < test_year
        test_mask = frame.index.year == test_year
        target_dates = pd.to_datetime(
            frame[target_date_column],
            errors="coerce",
        )
        mature_train_mask = train_mask & (target_dates <= test_start)
        known_train_mask = mature_train_mask & frame[target_column].notna()
        if int(known_train_mask.sum()) < min_train_samples:
            continue

        x_train = frame.loc[known_train_mask, feature_columns]
        x_test = frame.loc[test_mask, feature_columns]
        y_train_return = frame.loc[known_train_mask, target_column].astype(float)

        return_model = return_model_factory()
        return_model.fit(x_train, y_train_return)

        fold = frame.loc[test_mask].copy()
        fold[TEST_YEAR_COLUMN] = fold.index.year
        fold[PREDICTED_RETURN_COLUMN] = return_model.predict(x_test)

        probability_train_mask = mature_train_mask & frame[probability_target_column].notna()
        y_train_probability = frame.loc[probability_train_mask, probability_target_column].astype(int)
        probability_model = probability_model_factory()
        class_count = y_train_probability.nunique()
        can_fit = (
            class_count >= 2
            or (class_count == 1 and getattr(probability_model, "supports_single_class", False))
            or (class_count == 0 and getattr(probability_model, "supports_empty_training", False))
        )
        if can_fit:
            probability_model.fit(frame.loc[probability_train_mask, feature_columns], y_train_probability)
            fold[PREDICTED_PROBABILITY_COLUMN] = _positive_class_probability(
                probability_model,
                x_test,
            )
        else:
            fold[PREDICTED_PROBABILITY_COLUMN] = np.nan

        fold[RANK_COLUMN] = (
            fold.groupby(level=0, sort=False)[PREDICTED_RETURN_COLUMN]
            .rank(ascending=False, method="average")
            .to_numpy(dtype=float)
        )
        predictions.append(fold)

    if not predictions:
        return frame.iloc[0:0].assign(
            **{
                TEST_YEAR_COLUMN: pd.Series(dtype="int64"),
                PREDICTED_RETURN_COLUMN: pd.Series(dtype="float64"),
                PREDICTED_PROBABILITY_COLUMN: pd.Series(dtype="float64"),
                RANK_COLUMN: pd.Series(dtype="float64"),
            }
        )

    return pd.concat(predictions).sort_index()


def summarize_backtest(
    predictions: pd.DataFrame,
    target_column: str = RETURN_TARGET_COLUMN,
    predicted_return_column: str = PREDICTED_RETURN_COLUMN,
    probability_target_column: str = PROBABILITY_TARGET_COLUMN,
    predicted_probability_column: str = PREDICTED_PROBABILITY_COLUMN,
) -> pd.Series:
    _require_columns(predictions, [target_column, predicted_return_column])

    actual_return = predictions[target_column]
    predicted_return = predictions[predicted_return_column]

    summary = {
        "samples": int((actual_return.notna() & predicted_return.notna()).sum()),
        "mae": mean_absolute_error(actual_return, predicted_return),
        "directional_accuracy": directional_accuracy(actual_return, predicted_return),
        "r2": r2_score(actual_return, predicted_return),
        "correlation": correlation(actual_return, predicted_return),
    }

    if (
        probability_target_column in predictions.columns
        and predicted_probability_column in predictions.columns
    ):
        actual_probability_target = predictions[probability_target_column]
        predicted_probability = predictions[predicted_probability_column]
        valid = actual_probability_target.notna() & predicted_probability.notna()
        actual_probability_target = actual_probability_target.loc[valid]
        predicted_probability = predicted_probability.loc[valid]
        predicted_label = predicted_probability >= 0.5

        summary.update(
            {
                "classification_samples": int(valid.sum()),
                "classification_accuracy": binary_accuracy(
                    actual_probability_target,
                    predicted_label,
                ),
                "precision": precision_score(actual_probability_target, predicted_label),
                "recall": recall_score(actual_probability_target, predicted_label),
                "brier_score": brier_score(
                    actual_probability_target,
                    predicted_probability,
                ),
                "roc_auc": roc_auc_score(
                    actual_probability_target,
                    predicted_probability,
                ),
            }
        )

    return pd.Series(summary)
