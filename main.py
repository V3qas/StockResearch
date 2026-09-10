from __future__ import annotations

import argparse

import pandas as pd

from src.backtest import (
    run_walk_forward_backtest,
    summarize_backtest,
)
from src.config import (
    BENCHMARK,
    FORECAST_DAYS,
    MAX_PRICE_AGE_DAYS,
    REFERENCE_CALENDAR,
    RUNS_DIR,
    START_DATE,
    TICKERS,
)
from src.datasets import (
    build_ticker_samples,
    combine_ticker_samples,
    save_combined_samples,
    save_ticker_samples,
)
from src.data_loader import effective_end_date, get_stock, ticker_to_filename
from src.calendars import benchmark_coverage, reference_sessions
from src.experiments import ExperimentRun
from src.features import FEATURE_COLUMNS
from src.research import default_model_specs, run_model_comparison


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
    parser.add_argument("--end-date", help="Exclusive snapshot date, YYYY-MM-DD; defaults to today in UTC.")
    parser.add_argument("--max-price-age-days", type=int, default=MAX_PRICE_AGE_DAYS)
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
    parser.add_argument(
        "--cross-sectional-backtest",
        action="store_true",
        help="Run one global walk-forward model and rank tickers per monthly decision date.",
    )
    parser.add_argument("--compare-models", action="store_true", help="Compare XGBoost, zero alpha, historical mean, Ridge/logistic and momentum.")
    parser.add_argument("--min-ranking-tickers", type=int, default=3)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--min-train-samples", type=int, default=24)
    args = parser.parse_args()
    try:
        args.end_date = effective_end_date(args.end_date)
        if pd.Timestamp(args.start_date) >= pd.Timestamp(args.end_date):
            raise ValueError("start-date must precede end-date.")
        if args.max_price_age_days < 0 or args.min_train_years < 1 or args.min_train_samples < 1 or args.min_ranking_tickers < 2:
            raise ValueError("Invalid age, training or ranking limits.")
    except ValueError as error:
        parser.error(str(error))
    return args


def format_percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.2%}"


def print_sample_table(samples: pd.DataFrame, limit: int) -> None:
    if samples.empty:
        print("No feature-eligible samples yet.")
        return

    if limit <= 0:
        return
    view = samples.tail(limit).copy()
    view["as_of"] = view.index.strftime("%Y-%m-%d")
    view["target_date"] = pd.to_datetime(view["future_target_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    view["price"] = view["Close"].map(lambda value: f"{value:,.2f}")
    view["price_plus_12m"] = view["future_price_12m"].map(
        lambda value: "n/a" if pd.isna(value) else f"{value:,.2f}"
    )
    view["return_12m"] = view["future_return_12m"].map(format_percent)
    view["benchmark_12m"] = view["benchmark_return_12m"].map(format_percent)
    view["alpha_12m"] = view["alpha_12m"].map(format_percent)

    print(
        view[
            [
                "as_of",
                "ticker",
                "label_status",
                "price",
                "target_date",
                "price_plus_12m",
                "return_12m",
                "benchmark_12m",
                "alpha_12m",
            ]
        ].to_string(index=False)
    )


def print_backtest_summary(summary: pd.Series) -> None:
    formats = {
        "samples": "{:.0f}",
        "classification_samples": "{:.0f}",
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
    benchmark_prices: pd.DataFrame,
    run: ExperimentRun,
    reference_dates: pd.DatetimeIndex,
    end_date: str,
    max_price_age_days: int = MAX_PRICE_AGE_DAYS,
) -> pd.DataFrame:
    print(f"\nLoading {ticker}...")
    prices = get_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        force_download=force_download,
    )

    print(
        "Data available: "
        f"{prices.index.min().date()} -> {prices.index.max().date()} "
        f"({len(prices):,} rows)"
    )

    prices.to_parquet(run.raw / f"stock_{ticker_to_filename(ticker)}.parquet")
    samples = build_ticker_samples(
        ticker=ticker,
        prices=prices,
        benchmark_prices=benchmark_prices,
        forecast_days=forecast_days,
        snapshot_date=pd.Timestamp(effective_end_date(end_date)) - pd.Timedelta(days=1),
        max_price_age_days=max_price_age_days,
        reference_dates=reference_dates,
    )
    samples["benchmark"] = BENCHMARK
    samples["return_definition"] = "adjusted_price_return_local_currency"
    output_path = save_ticker_samples(samples, ticker=ticker, output_dir=run.processed)

    print(f"Feature-eligible samples: {len(samples):,}; labels: {samples.label_status.value_counts().to_dict()}")
    print(f"Saved: {output_path}")
    print_sample_table(samples, limit=sample_count)

    if not run_backtest:
        return samples

    backtest_predictions = run_walk_forward_backtest(
        samples,
        min_train_years=min_train_years,
        min_train_samples=min_train_samples,
    )
    backtest_path = (
        run.predictions / f"{ticker_to_filename(ticker)}_backtest_predictions.parquet"
    )
    backtest_predictions.to_parquet(backtest_path)

    if backtest_predictions.empty:
        print("\nNo backtest predictions generated; saved an empty result.")
        return samples

    print_backtest_summary(summarize_backtest(backtest_predictions))
    print(f"Backtest predictions saved: {backtest_path}")
    return samples


def main() -> None:
    args = parse_args()
    tickers = args.tickers or TICKERS
    if len({ticker_to_filename(ticker) for ticker in tickers}) != len(tickers):
        raise ValueError("Provide unique tickers with distinct cache filenames.")
    comparison_requested = bool(args.cross_sectional_backtest or args.compare_models)
    specs = default_model_specs() if comparison_requested else []
    if specs and not args.compare_models:
        specs = specs[:1]
    metadata = {
        "start_date": args.start_date,
        "end_date_exclusive": args.end_date,
        "tickers": tickers,
        "benchmark": BENCHMARK,
        "reference_calendar": REFERENCE_CALENDAR,
        "forecast_days": FORECAST_DAYS,
        "max_price_age_days": args.max_price_age_days,
        "min_train_years": args.min_train_years,
        "min_train_samples": args.min_train_samples,
        "min_ranking_tickers": args.min_ranking_tickers,
        "comparison_requested": comparison_requested,
        "models": [spec.name for spec in specs],
        "per_ticker_backtest_requested": bool(args.backtest),
        "force_download": bool(args.force_download),
        "return_definition": "adjusted_price_return_local_currency",
        "feature_columns": FEATURE_COLUMNS,
    }
    snapshot_date = pd.Timestamp(args.end_date) - pd.Timedelta(days=1)
    with ExperimentRun(RUNS_DIR, metadata) as run:
        print(f"Run: {run.run_id}")
        sessions = reference_sessions(pd.Timestamp(args.start_date), snapshot_date, FORECAST_DAYS)
        pd.DataFrame(index=sessions.rename("session")).to_parquet(run.raw / "reference_calendar.parquet")
        print(f"Loading benchmark {BENCHMARK}...")
        benchmark_prices = get_stock(
            ticker=BENCHMARK, start_date=args.start_date,
            end_date=args.end_date, force_download=args.force_download,
        )
        benchmark_prices.to_parquet(run.raw / "benchmark.parquet")
        coverage = benchmark_coverage(benchmark_prices, sessions, pd.Timestamp(args.start_date), snapshot_date)
        coverage.to_csv(run.processed / "benchmark_coverage.csv")
        print(f"Benchmark sessions without valid close: {(~coverage.available).sum()} / {len(coverage)}")

        sample_frames = []
        for ticker in tickers:
            sample_frames.append(process_ticker(
                ticker=ticker,
                start_date=args.start_date,
                forecast_days=FORECAST_DAYS,
                force_download=args.force_download,
                sample_count=args.samples,
                run_backtest=args.backtest,
                min_train_years=args.min_train_years,
                min_train_samples=args.min_train_samples,
                benchmark_prices=benchmark_prices,
                run=run,
                reference_dates=sessions,
                end_date=args.end_date,
                max_price_age_days=args.max_price_age_days,
            ))
        combined_samples = combine_ticker_samples(sample_frames)
        combined_path = save_combined_samples(combined_samples, output_dir=run.processed)
        print(f"\nCombined dataset: {len(combined_samples):,} rows, {combined_samples.ticker.nunique():,} tickers")
        print(f"Saved: {combined_path}")

        if comparison_requested:
            comparison = run_model_comparison(
                combined_samples,
                min_train_years=args.min_train_years,
                min_train_samples=args.min_train_samples,
                min_ranking_tickers=args.min_ranking_tickers,
                model_specs=specs,
            )
            predictions = comparison.predictions
            predictions.loc[predictions.model == "xgboost"].to_parquet(
                run.predictions / "cross_sectional_backtest_predictions.parquet",
            )
            predictions.to_parquet(run.predictions / "model_comparison_predictions.parquet")
            comparison.summary.to_csv(run.predictions / "model_comparison_summary.csv")
            comparison.yearly_summary.to_csv(run.predictions / "model_comparison_by_year.csv")
            comparison.ranking_by_date.to_parquet(run.predictions / "ranking_metrics_by_date.parquet")
            comparison.classification_comparison.to_csv(run.predictions / "classification_common_cohort.csv")
            if predictions.empty:
                print("\nNo cross-sectional predictions generated; saved empty results.")
            columns = ["eligible_samples", "observed_labels", "samples", "mae", "ic_dates", "mean_rank_ic"]
            print("\nModel comparison (MAE in return units; IC uses complete dates)")
            print(comparison.summary[columns].to_string(float_format=lambda value: f"{value:.4f}"))
            print("\nClassification comparison (same observed rows for every classifier)")
            print(comparison.classification_comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    print(f"\nCompleted run: {run.path}")
    print(f"Latest successful run: {RUNS_DIR / 'latest.json'}")


if __name__ == "__main__":
    main()
