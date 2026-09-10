# StockResearch

V0.1 price-data pipeline for historical 12-month return samples.

The first milestone is intentionally small:

1. Download adjusted daily price data with `yfinance`.
2. Cache raw prices as parquet files under `data/raw/`.
3. Add basic momentum, moving-average, and volatility features.
4. Build point-in-time 252-trading-day forward return targets.
5. Save monthly training samples under `data/processed/`.

## Setup

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Process the default tickers from `src/config.py`:

```powershell
.\.venv\Scripts\python.exe main.py
```

Process one ticker:

```powershell
.\.venv\Scripts\python.exe main.py TSLA
```

Force a fresh Yahoo Finance download instead of cached parquet data:

```powershell
.\.venv\Scripts\python.exe main.py TSLA --force-download
```

Build samples and run the first walk-forward model backtest:

```powershell
.\.venv\Scripts\python.exe main.py TSLA --backtest
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Next Milestone

Add benchmark data and compute `alpha_12m` next to `future_return_12m`.
