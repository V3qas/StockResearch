# StockResearch

Research pipeline for monthly stock selection from price-based momentum and
volatility features. It predicts benchmark-relative returns and outperformance
probabilities, then compares XGBoost with simple baselines on historical
walk-forward folds.

## Setup and tests

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

## Run

Run the **Does StockResearch find signal?** experiment with 300 US stocks:

```powershell
.\.venv\Scripts\python.exe main.py --universe-file universes/us_large_cap_300.json --compare-models --start-date 2012-01-01 --end-date 2026-09-10 --min-ranking-tickers 100 --samples 0
```

The frozen universe selects the largest 300 equity positions by holding market
value from the official [iShares IVV holdings](https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/latest-holdings.csv)
dated September 9, 2026, restricted to US location, USD and US exchanges. Share
classes count separately. The JSON records the selection rule, original-source
hash and limitations. This is an **exploratory current-survivor universe**, not
point-in-time S&P 500 membership. Do not interpret a positive result as an
unbiased historical signal. Positional tickers and `--universe-file` are mutually
exclusive; the five-ticker default remains useful for quick checks.

The report is saved to `data/runs/<run_id>/predictions/signal_report.md`, with
`signal_diagnostics.png` for annual IC and decile curves. It includes mean and
median Spearman rank IC, the share of defined IC months above zero, MAE, annual
coverage, paired XGBoost/baseline IC differences, and realized forward excess
returns for ten score buckets. The five-year default training window means that
2012 price history produces test predictions starting in **2017**; recent
predictions with unmatured outcomes remain pending, including all of 2026.

Decile membership uses score ranks at each decision date, before outcomes are
examined. Average-rank ties remain together; this can make buckets unequal or
empty. A reported curve uses complete dates with all ten buckets populated and
at least `max(10, min_ranking_tickers)` stocks. Missing outcomes suppress the
entire date's decile returns; assignments and coverage remain available in
`rank_decile_assignments.parquet` and `decile_returns_by_date.csv`. Per-ticker
feature and label counts are saved in `universe_coverage.csv`. Dates receive
equal weight in `decile_summary.csv`. The pooled historical mean is constant
within each month, so it has no ranking IC or decile curve. Momentum has no MAE
because its raw ranking score is not a calibrated alpha forecast.

`price_jump_audit.csv` flags adjacent adjusted-close changes of at least +100%
or at most -50%. No prices or stocks are automatically removed. Review flags
before drawing conclusions. Monthly labels overlap: no significance, strategy
profitability or independent-observation count is claimed.

Re-fit from a run's verified input snapshots **without any downloads**:

```powershell
.\.venv\Scripts\python.exe replay_signal.py data/runs/<run_id>
```

Replay checks the saved raw/processed input hashes and writes a new isolated
run with a reference to its parent. It uses the current source and environment;
for exact numerical reproduction retain the recorded source and dependency
versions as well. A changed model implementation is a new experiment. Original
inputs and reports remain available in their own run directory.

Build samples for the five default tickers:

```powershell
.\.venv\Scripts\python.exe main.py
```

Run the global model and all baselines on an explicit snapshot:

```powershell
.\.venv\Scripts\python.exe main.py --compare-models --end-date 2026-09-10
```

Start the local dashboard after a successful run:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The dashboard opens in a browser and reads only the latest successful run from
`data/runs/latest.json` on every rerun. It does not retrain models on page load.

The dashboard contains an overview with key metrics and latest signals, a
model comparison with annual charts, a selectable ticker analysis with price
history, ranking-quality metrics and a data-quality/run-details view. Missing
optional report artifacts are shown as unavailable instead of breaking the
page.

Other supported workflows:

```powershell
.\.venv\Scripts\python.exe main.py TSLA --backtest
.\.venv\Scripts\python.exe main.py --cross-sectional-backtest
.\.venv\Scripts\python.exe main.py --compare-models --min-ranking-tickers 4
.\.venv\Scripts\python.exe main.py --force-download
```

- `--end-date` is exclusive and defaults to today's UTC date. Daily bars from
  today are excluded to avoid treating an unfinished session as a close.
- All tickers and the benchmark use that same snapshot boundary.
- Raw caches include the resolved start/end dates and adjustment convention.
  A new UTC day, changed request, missing/invalid metadata or
  `--force-download` causes a download. An explicit historical snapshot can be
  reused; later provider revisions still require a fresh download.
- `--max-price-age-days` defaults to 7 calendar days.
- `--min-train-years` and `--min-train-samples` default to 5 and 24.
- The CLI horizon is fixed at 252 reference sessions; the ambiguous
  `--forecast-days` option has been removed.

## Dataset contract

1. `as_of` is a common calendar month end, interpreted after all covered
   exchanges have closed. It is selected **before** checking features or labels.
2. `price_date` records the actual stock observation used at that decision date.
   The last observation at or before `as_of` is eligible only within the age
   limit. Missing features never cause selection of an earlier observation.
3. `future_target_date` is the 252nd **XNYS exchange session** strictly after
   `as_of`, using the independent, version-pinned `exchange-calendars` package.
   All tickers at that `as_of` have the same planned target date, including
   samples whose outcomes have not matured. Missing benchmark quote rows cannot
   move the horizon. The exact calendar used is saved with every run.
4. `future_price_date`, `benchmark_price_date` and
   `benchmark_future_price_date` retain the actual observations used for returns.
   Stock lookups use the bounded preceding quote. Benchmark lookups require
   the exact expected exchange-session close at or before the requested date,
   also within the age limit. A missing expected close yields an unknown label,
   even if another close is available just one day earlier. No later quote is used.
5. Eligibility depends only on the stock price and features available at `as_of`.
   Missing future outcomes **never remove a ticker from predictions or ranks**.
   `label_status` distinguishes `observed`, `pending` (target beyond the snapshot)
   and `missing` (mature target with insufficient prices). Appending future data
   fills outcomes without changing existing features, decision dates or planned
   target dates, provided historical prices and the calendar version are unchanged.
   A pending target is never filled from a recent quote before its maturity.
6. Legacy `*_12m` columns refer to the fixed 252-session horizon, approximately
   one year. `forecast_days` is also stored in the dataset.

The benchmark is **SPY with adjusted prices**, providing a dividend-adjusted
S&P 500 ETF return proxy with fund expenses. Stocks also use adjusted prices.
This replaces the previous comparison against the S&P 500 price index
(`^GSPC`), which omitted index dividends.

`alpha_12m = future_return_12m - benchmark_return_12m` is a simple excess
return, **not risk-adjusted factor alpha**. Returns are still measured in each
security's local currency: RHM.DE versus a USD benchmark remains a research
simplification until currency conversion and market-specific benchmarks exist.

## Validation and baselines

The expanding walk-forward fits new models once per test year. Training uses
only earlier samples with an observed training target and
`future_target_date <= first test as_of`.
Features for each later test month become available at that month's decision.
Per-ticker and global backtests share this logic.

`--compare-models` evaluates:

| Model | Alpha forecast | Outperformance probability |
| --- | --- | --- |
| XGBoost | XGBRegressor | XGBClassifier |
| Zero alpha | 0 | 50% |
| Historical mean | Mean of mature training labels | Training outperformance rate |
| Ridge / logistic | StandardScaler + Ridge | StandardScaler + LogisticRegression |
| Momentum | 120-session return used only as a ranking score | Unavailable |

All models use identical dated test rows; scalers and historical averages are
fitted only inside each training fold. No test-set tuning is performed.
XGBoost and logistic regression leave probabilities unavailable if a training
fold has fewer than two classes. The constant 50% baseline remains defined;
historical frequency also works with one class, returning 0% or 100%, but needs
at least one observed training label. All models still share the same regression
training thresholds and test folds.

Classification metrics count only pairs with both a target and a probability.
Regression and classification sample counts and label coverage are reported
separately. `classification_common_cohort.csv` compares classifiers only on the
intersection of rows with an observed label and valid probability from **every
classifier**. Use this file for classification model comparisons; the general
summary and annual report also retain each model's own available-row metrics.
Momentum
is evaluated only for ranking; its score is not presented as predicted alpha.

Rankings are computed per decision date. Ties share an average rank.
Per-date IC (Pearson) and rank IC (Spearman) require at least three tickers by
default, complete outcomes and scores for the entire eligible cross-section,
and variation in both scores and outcomes. Partial dates retain universe counts
and label coverage, but their IC is unavailable. Constant forecasts have
undefined IC rather than a ticker-order signal. Reports include ticker counts,
eligible IC-date counts, equal-weight mean IC across dates, and results by year.

Monthly forward returns overlap. Counts are not counts of independent
observations; the reports do not claim statistical significance. These are
prediction backtests, not executable portfolio simulations with transaction
costs and rebalancing.

## Outputs

`data/raw/<ticker>.parquet` and `.json` remain reusable download caches.
Every invocation writes a unique directory under `data/runs/<run_id>/`:

- `raw/stock_<ticker>.parquet`, `raw/benchmark.parquet`: copies of the actual
  input frames, unaffected by later cache refreshes.
- `raw/reference_calendar.parquet`: reference sessions, including future targets.
- `processed/stock_<ticker>_samples.parquet` and `processed/all_samples.parquet`:
  all feature-eligible rows, including unknown outcomes, sorted by date/ticker.
  The stock prefix prevents the real `ALL` ticker from overwriting the combined
  dataset on case-insensitive filesystems.
- `processed/benchmark_coverage.csv`: available/missing expected benchmark closes.
- `predictions/<ticker>_backtest_predictions.parquet`: optional per-ticker results.
- `predictions/cross_sectional_backtest_predictions.parquet`: global XGBoost results.
- `predictions/model_comparison_predictions.parquet`: all model outputs and ranking scores.
- `predictions/model_comparison_summary.csv` and `model_comparison_by_year.csv`:
  aggregate and annual metrics with coverage counts.
- `predictions/classification_common_cohort.csv`: classification metrics on shared rows.
- `predictions/ranking_metrics_by_date.parquet`: IC, universe counts and label coverage.
- `source/`: copies of the pipeline source and dependency specification.
- `manifest.json`: status, snapshot, universe, parameters, dependency versions
  and SHA-256 hashes of the saved inputs, code and results.

Only after all requested outputs succeed is `data/runs/latest.json` replaced
atomically. Resolve it once, then read files from that run:

```python
import json
from pathlib import Path
import pandas as pd

runs = Path("data/runs")
run = runs / json.loads((runs / "latest.json").read_text())["run_id"]
summary = pd.read_csv(run / "predictions/model_comparison_summary.csv")
```

A sample-only run has no comparison files; its manifest records which operations
were requested. Failed or interrupted runs stay in separate directories and
never replace the last successful pointer. A completed run with no eligible
backtest folds saves explicit empty results. The old shared `data/processed/`
and `data/predictions/` files are legacy outputs and are no longer updated.

An explicit date alone does not freeze provider revisions: retain the run's
input snapshots for reproduction. Calendar upgrades or newly announced exchange
closures can change future planned dates; the stored calendar and package
version identify the convention used. The manifest records the complete installed
Python package set, but most versions are not yet pinned in `requirements.txt`.

Unknown delisting outcomes are reported, not imputed. Metrics remain conditional
on observed outcomes (and IC on complete dates). A fixed, manually selected
ticker list does not resolve survivorship bias; historical universe membership
and reliable delisting returns remain necessary for broader research claims.

## Next steps

Evaluate baseline and ranking stability on a larger, explicitly defined
universe; harmonize currencies and market benchmarks; quantify uncertainty for
overlapping labels. Model tuning and persisted live rankings follow after a
repeatable advantage has been demonstrated.
