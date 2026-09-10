from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"

TICKERS = [
    "TSLA",
    "BABA",
    "TEAM",
    "ICHR",
    "RHM.DE",
]

BENCHMARK = "^GSPC"
START_DATE = "2010-01-01"
FORECAST_DAYS = 252
