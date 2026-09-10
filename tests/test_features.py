from __future__ import annotations

import pandas as pd
import pytest

from src.features import add_features


def test_add_features_creates_expected_columns() -> None:
    dates = pd.bdate_range("2020-01-01", periods=260)
    df = pd.DataFrame({"Close": range(100, 360)}, index=dates)

    result = add_features(df)

    assert "return_20d" in result.columns
    assert "distance_sma_200" in result.columns
    assert "volatility_60d" in result.columns
    assert result["return_20d"].iloc[20] == pytest.approx(20 / 100)
    assert pd.notna(result["distance_sma_200"].iloc[199])
