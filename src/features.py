from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "return_120d",
    "distance_sma_20",
    "distance_sma_50",
    "distance_sma_200",
    "volatility_20d",
    "volatility_60d",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Expected a Close column.")

    featured = df.copy()
    close = pd.to_numeric(featured["Close"], errors="coerce")

    featured["return_5d"] = close.pct_change(5, fill_method=None)
    featured["return_20d"] = close.pct_change(20, fill_method=None)
    featured["return_60d"] = close.pct_change(60, fill_method=None)
    featured["return_120d"] = close.pct_change(120, fill_method=None)

    featured["sma_20"] = close.rolling(20).mean()
    featured["sma_50"] = close.rolling(50).mean()
    featured["sma_200"] = close.rolling(200).mean()

    featured["distance_sma_20"] = close / featured["sma_20"] - 1
    featured["distance_sma_50"] = close / featured["sma_50"] - 1
    featured["distance_sma_200"] = close / featured["sma_200"] - 1

    daily_return = close.pct_change(fill_method=None)
    featured["volatility_20d"] = daily_return.rolling(20).std()
    featured["volatility_60d"] = daily_return.rolling(60).std()

    return featured
