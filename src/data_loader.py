from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR


STANDARD_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def effective_end_date(end_date: str | None = None) -> str:
    """Exclusive UTC snapshot boundary; never cache an unfinished daily bar."""
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    boundary = today if end_date is None else pd.Timestamp(end_date)
    if boundary.tz is not None or boundary != boundary.normalize() or boundary > today:
        raise ValueError("end_date must be a date no later than today (exclusive).")
    return boundary.date().isoformat()


def ticker_to_filename(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).strip("_")
    return cleaned or "ticker"


def raw_data_path(ticker: str) -> Path:
    return RAW_DATA_DIR / f"{ticker_to_filename(ticker)}.parquet"


def raw_data_metadata_path(ticker: str) -> Path:
    return raw_data_path(ticker).with_suffix(".json")


def _cache_metadata(
    start_date: str,
    end_date: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "start_date": start_date,
        "end_date": effective_end_date(end_date),
        "auto_adjust": True,
    }


def _cache_matches_request(
    ticker: str,
    start_date: str,
    end_date: str | None,
) -> bool:
    metadata_path = raw_data_metadata_path(ticker)
    if not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return metadata == _cache_metadata(start_date, end_date)


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
    if normalized.index.has_duplicates or normalized.index.hasnans:
        raise ValueError(f"Duplicate or missing price dates for {ticker}.")

    available_columns = [
        column for column in STANDARD_PRICE_COLUMNS if column in normalized.columns
    ]
    if "Close" not in available_columns:
        raise ValueError(f"No Close column found for {ticker}.")

    normalized = normalized[available_columns]
    normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
    normalized = normalized.loc[np.isfinite(normalized.Close) & (normalized.Close > 0)]
    if normalized.empty:
        raise ValueError(f"No valid prices returned for {ticker}.")

    return normalized


def download_stock(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    import yfinance as yf

    end_date = effective_end_date(end_date)
    if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
        raise ValueError("start_date must be earlier than end_date.")
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
    normalized = normalized.loc[
        (normalized.index >= pd.Timestamp(start_date))
        & (normalized.index < pd.Timestamp(end_date))
    ]
    if normalized.empty:
        raise ValueError(f"No prices in the requested snapshot for {ticker}.")
    normalized.to_parquet(raw_data_path(ticker))
    raw_data_metadata_path(ticker).write_text(
        json.dumps(_cache_metadata(start_date, end_date), indent=2),
        encoding="utf-8",
    )

    return normalized


def load_stock(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(raw_data_path(ticker))


def get_stock(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    end_date = effective_end_date(end_date)
    path = raw_data_path(ticker)
    if path.exists() and not force_download and _cache_matches_request(
        ticker,
        start_date,
        end_date,
    ):
        return load_stock(ticker)

    return download_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
