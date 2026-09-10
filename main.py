from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import run_walk_forward_backtest, summarize_backtest
from src.config import (
    FORECAST_DAYS,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    START_DATE,
    TICKERS,
)
from src.data_loader import get_stock, ticker_to_filename
from src.features import add_features
from src.targets import add_targets, build_monthly_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build first historical 12M stock samples from price data."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker symbols to process, e.g. TSLA BABA RHM.DE.",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--forecast-days", type=int, default=FORECAST_DAYS)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached raw parquet files and download again.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of latest monthly samples to print per ticker.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run an expanding yearly walk-forward backtest after building samples.",
    )
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--min-train-samples", type=int, default=24)
    return parser.parse_args()


def format_percent(value: float) -> str:
    return f"{value:+.2%}"


def print_sample_table(samples: pd.DataFrame, limit: int) -> None:
    if samples.empty:
        print("No complete 12M samples yet.")
        return

    view = samples.tail(limit).copy()
    view["as_of"] = view.index.strftime("%Y-%m-%d")
    view["target_date"] = pd.to_datetime(view["future_target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    view["price"] = view["Close"].map(lambda value: f"{value:,.2f}")
    view["price_plus_12m"] = view["future_price_12m"].map(
        lambda value: f"{value:,.2f}"
    )
    view["return_12m"] = view["future_return_12m"].map(format_percent)

    print(
        view[
            ["as_of", "price", "target_date", "price_plus_12m", "return_12m"]
        ].to_string(index=False)
    )


def print_backtest_summary(summary: pd.Series) -> None:
    formats = {
        "samples": "{:.0f}",
        "mae": "{:.2%}",
        "directional_accuracy": "{:.1%}",
        "r2": "{:.3f}",
        "correlation": "{:.3f}",
        "classification_accuracy": "{:.1%}",
        "precision": "{:.1%}",
        "recall": "{:.1%}",
        "brier_score": "{:.4f}",
        "roc_auc": "{:.3f}",
    }

    print("\nBacktest summary")
    for metric, template in formats.items():
        if metric not in summary:
            continue

        value = summary[metric]
        formatted = "n/a" if pd.isna(value) else template.format(value)
        print(f"{metric:>24}: {formatted}")


def process_ticker(
    ticker: str,
    start_date: str,
    forecast_days: int,
    force_download: bool,
    sample_count: int,
    run_backtest: bool,
    min_train_years: int,
    min_train_samples: int,
) -> None:
    print(f"\nDownloading {ticker}...")
    prices = get_stock(
        ticker=ticker,
        start_date=start_date,
        force_download=force_download,
    )

    print(
        "Data available: "
        f"{prices.index.min().date()} -> {prices.index.max().date()} "
        f"({len(prices):,} rows)"
    )

    with_features = add_features(prices)
    with_targets = add_targets(with_features, forecast_days=forecast_days)
    samples = build_monthly_samples(with_targets)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"{ticker_to_filename(ticker)}_samples.parquet"
    samples.to_parquet(output_path)

    print(f"Generated historical samples: {len(samples):,}")
    print(f"Saved: {output_path}")
    print_sample_table(samples, limit=sample_count)

    if not run_backtest:
        return

    backtest_predictions = run_walk_forward_backtest(
        samples,
        min_train_years=min_train_years,
        min_train_samples=min_train_samples,
    )
    if backtest_predictions.empty:
        print("\nNo backtest predictions generated.")
        return

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    backtest_path = (
        PREDICTIONS_DIR / f"{ticker_to_filename(ticker)}_backtest_predictions.parquet"
    )
    backtest_predictions.to_parquet(backtest_path)

    print_backtest_summary(summarize_backtest(backtest_predictions))
    print(f"Backtest predictions saved: {backtest_path}")


def main() -> None:
    args = parse_args()
    tickers = args.tickers or TICKERS

    for ticker in tickers:
        process_ticker(
            ticker=ticker,
            start_date=args.start_date,
            forecast_days=args.forecast_days,
            force_download=args.force_download,
            sample_count=args.samples,
            run_backtest=args.backtest,
            min_train_years=args.min_train_years,
            min_train_samples=args.min_train_samples,
        )


if __name__ == "__main__":
    main()
