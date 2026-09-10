from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR


STANDARD_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def ticker_to_filename(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).strip("_")
    return cleaned or "ticker"


def raw_data_path(ticker: str) -> Path:
    return RAW_DATA_DIR / f"{ticker_to_filename(ticker)}.parquet"


def normalize_price_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    normalized = df.copy()

    if isinstance(normalized.columns, pd.MultiIndex):
        last_level = normalized.columns.get_level_values(-1).astype(str)
        first_level = normalized.columns.get_level_values(0).astype(str)

        if ticker in set(last_level):
            normalized = normalized.xs(ticker, axis=1, level=-1, drop_level=True)
        elif len(set(last_level)) == 1:
            normalized.columns = normalized.columns.droplevel(-1)
        elif len(set(first_level)) == 1:
            normalized.columns = normalized.columns.droplevel(0)
        else:
            normalized.columns = [
                "_".join(str(part) for part in column if str(part))
                for column in normalized.columns
            ]

    normalized.index = pd.to_datetime(normalized.index)
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_convert(None)

    normalized = normalized.sort_index()
    normalized.index.name = "Date"

    available_columns = [
        column for column in STANDARD_PRICE_COLUMNS if column in normalized.columns
    ]
    if "Close" not in available_columns:
        raise ValueError(f"No Close column found for {ticker}.")

    normalized = normalized[available_columns]
    normalized = normalized.dropna(subset=["Close"])

    return normalized


def download_stock(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    import yfinance as yf

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    normalized = normalize_price_data(df, ticker)
    normalized.to_parquet(raw_data_path(ticker))

    return normalized


def load_stock(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(raw_data_path(ticker))


def get_stock(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    path = raw_data_path(ticker)
    if path.exists() and not force_download:
        return load_stock(ticker)

    return download_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
