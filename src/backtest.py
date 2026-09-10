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


def _known_training_index(
    df: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    target_date_column: str,
) -> pd.Index:
    if target_date_column not in df.columns or len(test_index) == 0:
        return train_index

    test_start = pd.Timestamp(test_index.min())
    target_dates = pd.to_datetime(
        df.loc[train_index, target_date_column],
        errors="coerce",
    )
    known_target_mask = target_dates <= test_start

    return target_dates.index[known_target_mask]


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
    if not isinstance(samples.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if min_train_samples < 1:
        raise ValueError("min_train_samples must be at least 1.")

    required_columns = [*feature_columns, target_column, probability_target_column]
    _require_columns(samples, required_columns)

    frame = samples.sort_index().dropna(subset=required_columns).copy()
    if frame.empty:
        return frame.assign(
            **{
                TEST_YEAR_COLUMN: pd.Series(dtype="int64"),
                PREDICTED_RETURN_COLUMN: pd.Series(dtype="float64"),
                PREDICTED_PROBABILITY_COLUMN: pd.Series(dtype="float64"),
            }
        )

    if return_model_factory is None:
        return_model_factory = _default_return_model_factory
    if probability_model_factory is None:
        probability_model_factory = _default_probability_model_factory

    predictions = []
    for train_index, test_index in yearly_walk_forward_splits(
        frame,
        min_train_years=min_train_years,
    ):
        train_index = _known_training_index(
            frame,
            train_index=train_index,
            test_index=test_index,
            target_date_column=target_date_column,
        )
        if len(train_index) < min_train_samples:
            continue

        x_train = frame.loc[train_index, feature_columns]
        y_train_return = frame.loc[train_index, target_column].astype(float)
        x_test = frame.loc[test_index, feature_columns]

        fold = frame.loc[test_index].copy()
        fold[TEST_YEAR_COLUMN] = fold.index.year

        return_model = return_model_factory()
        return_model.fit(x_train, y_train_return)
        fold[PREDICTED_RETURN_COLUMN] = return_model.predict(x_test)

        y_train_probability = frame.loc[train_index, probability_target_column].astype(
            int
        )
        if y_train_probability.nunique() >= 2:
            probability_model = probability_model_factory()
            probability_model.fit(x_train, y_train_probability)
            fold[PREDICTED_PROBABILITY_COLUMN] = _positive_class_probability(
                probability_model,
                x_test,
            )
        else:
            fold[PREDICTED_PROBABILITY_COLUMN] = np.nan

        predictions.append(fold)

    if not predictions:
        return frame.iloc[0:0].assign(
            **{
                TEST_YEAR_COLUMN: pd.Series(dtype="int64"),
                PREDICTED_RETURN_COLUMN: pd.Series(dtype="float64"),
                PREDICTED_PROBABILITY_COLUMN: pd.Series(dtype="float64"),
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
        "samples": int(predictions[predicted_return_column].notna().sum()),
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
        predicted_label = predicted_probability >= 0.5

        summary.update(
            {
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
