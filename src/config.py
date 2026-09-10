from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = DATA_DIR / "runs"

TICKERS = [
    "TSLA",
    "BABA",
    "TEAM",
    "ICHR",
    "RHM.DE",
]

# Adjusted ETF prices include distributions, unlike the S&P 500 price index.
BENCHMARK = "SPY"
START_DATE = "2010-01-01"
FORECAST_DAYS = 252
MAX_PRICE_AGE_DAYS = 7
REFERENCE_CALENDAR = "XNYS"
