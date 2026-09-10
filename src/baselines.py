from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantReturnModel:
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(x_test))


class HistoricalMeanModel:
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.mean = float(y_train.mean())

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return np.full(len(x_test), self.mean)


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])
    supports_single_class = True
    supports_empty_training = True

    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(x_test), self.probability)
        return np.column_stack([1 - positive, positive])


class HistoricalFrequencyModel(ConstantProbabilityModel):
    supports_empty_training = False

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.probability = float(y_train.mean())


class MomentumRankingModel:
    """120-session momentum is a ranking score, not an alpha forecast."""

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        return x_test["return_120d"].to_numpy(dtype=float)


def create_ridge_model():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def create_logistic_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
