# Research status: Does StockResearch identify a robust stock-ranking signal?

As of September 11, 2026

## Executive summary

In the current reference experiment, StockResearch identifies a clear
cross-sectional ranking signal. The result is exploratory, however, and does
not yet establish a profitable or unbiased alpha signal going forward.

The most important refinement to the research findings so far is:

> The strong Ridge result is explained almost entirely by 60-day volatility.
> Momentum and SMA features contribute little independent ranking information
> in the current sample.

A Ridge regression using only `volatility_60d` achieves a mean rank IC of
`+0.1324`. The full nine-feature Ridge model achieves `+0.1297`. When both
volatility features are removed, mean rank IC falls to `+0.0148`.

The central research question is therefore no longer whether Ridge needs better
hyperparameters, but:

> Does the positive relationship between past 60-day volatility and subsequent
> relative returns survive in a genuine point-in-time universe that includes
> subsequent losers, index deletions, and delistings?

Before answering this question with new point-in-time data, the research
pipeline must first be validated adversarially on data with known ground truth.
Negative controls must produce null results, and positive controls must reliably
recover a predefined signal. This pipeline red-team test is a mandatory Step 0
before the point-in-time run.

## 1. Precise research questions

The project now examines three sequential hypotheses:

1. Do historical price features contain information about the future
   **relative ranking** of stocks?
2. Does a trained model provide incremental information beyond simple,
   transparent single-factor scores?
3. Does the result survive the removal of universe bias, controls for known
   factors, and a realistic treatment of time dependence?

The current results provide exploratory evidence for Hypothesis 1. The
reference experiment provides no evidence for Hypothesis 2 at present: a simple
volatility score reproduces Ridge. Hypothesis 3 remains open.

## 2. Reference experiment

All central claims in this document refer to the completed run
`20260910T221307_4dbef5caf234`.

| Property | Specification |
| --- | --- |
| Universe | 300 largest U.S. equity positions in the IVV snapshot dated September 9, 2026 |
| Membership | Current survivor universe applied retrospectively; not point-in-time |
| Price request | January 1, 2012 through September 10, 2026, with an exclusive end date |
| Benchmark | SPY adjusted price |
| Target | Stock return minus SPY over 252 XNYS sessions, approximately twelve months |
| Decision frequency | Monthly |
| Out-of-sample period | January 2017 through August 2026 |
| Fully observed target periods | January 2017 through August 2025 |
| Training | Expanding annual walk-forward with at least five prior calendar years |
| Minimum ranking breadth | 100 stocks per decision date |

Of the 300 requested stocks, 299 have feature samples; `HONA` has insufficient
history. The pipeline generated 47,507 feature samples in total. Each model
comparison contains 33,684 out-of-sample predictions, of which 30,098 have an
observed target. The ranking evaluation covers 104 months with fully observed
outcomes.

The experiment can be reproduced with the following command:

```powershell
.\.venv\Scripts\python.exe main.py --universe-file universes/us_large_cap_300.json --compare-models --start-date 2012-01-01 --end-date 2026-09-10 --min-ranking-tickers 100 --samples 0
```

## 3. Features actually used

The full model uses exactly nine technical price features:

- Returns over 5, 20, 60, and 120 trading days
- Distance from the 20-, 50-, and 200-day SMA
- Realized volatility over 20 and 60 trading days

The model currently uses no fundamental data, valuation metrics, analyst
estimates, sectors, market capitalizations, or macroeconomic data.

For each annual test fold, training uses only targets whose 252-session horizon
had ended by the first test date in that fold. Scaling and model fitting use
only those training rows.

## 4. Interpretation of the metrics

**Spearman rank IC** measures, for each monthly decision date, the rank
correlation between the score and subsequent excess return. Mean and median IC
give equal weight to every evaluable month.

**IC > 0** is the proportion of months with a positive Spearman rank
correlation.

**Top-bottom spread** is the difference between the average subsequent
12-month excess returns of the highest- and lowest-score deciles. It is a
forward-return diagnostic, not an annualized or directly tradable portfolio
return.

**MAE** evaluates the absolute calibration of predicted excess returns. A model
can have poor MAE while still producing a useful ranking.

## 5. Main model-comparison result

All models were compared on identical out-of-sample rows.

| Model | Mean rank IC | Median rank IC | IC > 0 | MAE | Top-bottom |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | +0.1084 | +0.1404 | 68.3% | 26.77 pp | +33.22% |
| Ridge | **+0.1297** | **+0.1452** | **71.2%** | 26.14 pp | **+42.64%** |
| 120-day momentum | +0.0149 | +0.0128 | 55.8% | undefined | +7.25% |
| Zero alpha | undefined | undefined | undefined | **25.66 pp** | undefined |
| Historical mean | undefined | undefined | undefined | 26.12 pp | undefined |

Ridge exceeds XGBoost's mean rank IC by `0.0212`. XGBoost leads Ridge in only
42.3% of their shared months and in three of nine evaluable years. XGBoost's
mean IC exceeds momentum by `0.0935`.

Neither trained model is convincing as an absolute return forecast: the
zero-alpha baseline has the lowest MAE. The ranking result must therefore not
be interpreted as a calibrated expected-return forecast.

## 6. Probability forecasts

The classification models have not yet demonstrated incremental value:

| Model | Brier score | ROC AUC |
| --- | ---: | ---: |
| 50% baseline | **0.2500** | 0.5000 |
| XGBoost | 0.2545 | 0.5307 |
| Ridge logistic | 0.2513 | 0.5450 |

The slightly higher AUC values are not enough to justify presenting poorly
calibrated probabilities as reliable `P(Outperformance)` estimates. These
probabilities should not currently be used in the product.

## 7. Stability by test year

| Year | XGBoost IC | XGBoost top-bottom | Ridge IC | Ridge top-bottom |
| --- | ---: | ---: | ---: | ---: |
| 2017 | +0.088 | +16.4% | +0.123 | +22.2% |
| 2018 | -0.130 | -6.1% | -0.137 | -4.8% |
| 2019 | +0.155 | +35.4% | +0.198 | +48.2% |
| 2020 | +0.318 | +49.3% | +0.415 | +74.2% |
| 2021 | -0.136 | -3.8% | -0.170 | -5.0% |
| 2022 | +0.158 | +26.9% | +0.168 | +25.5% |
| 2023 | +0.214 | +57.5% | +0.233 | +69.1% |
| 2024 | +0.095 | +41.5% | +0.083 | +51.6% |
| 2025 | +0.267 | +106.3% | +0.318 | +132.8% |

Only eight months in 2025 have fully observed outcomes. The unusually large
spreads in this partial year must not be treated as a full calendar-year result.
Excluding 2025, Ridge still has a mean rank IC of `+0.1139` and an average
top-bottom spread of `+35.12%`. The positive overall result is therefore not
driven solely by 2025.

The Ridge, XGBoost, and subsequently isolated volatility scores are all
negative in 2018 and 2021. This establishes instability over time, but it does
not identify a specific market regime or explain the cause.

## 8. Exact reconstruction of the Ridge models

The Ridge models for test years 2017 through 2025 were refitted using the same
training, label-maturity, and scaling rules. In every fold, the maximum absolute
difference between the reconstructed and stored predictions is exactly `0.0`.

The values below are coefficients on standardized features. They show the
average change in predicted excess return for a one-training-standard-deviation
increase in a feature, holding the other correlated features constant.

| Feature | Mean coefficient | Sign consistency |
| --- | ---: | ---: |
| `volatility_60d` | +0.0587 | positive in 9/9 folds |
| `return_120d` | +0.0392 | positive in 9/9 |
| `distance_sma_200` | -0.0345 | negative in 9/9 |
| `return_20d` | -0.0218 | negative in 9/9 |
| `return_60d` | +0.0173 | positive in 9/9 |
| `distance_sma_20` | +0.0164 | positive in 9/9 |
| `volatility_20d` | +0.0116 | positive in 9/9 |
| `return_5d` | -0.0080 | negative in 8/9 |
| `distance_sma_50` | +0.0055 | positive in 7/9 |

Because returns and distances from SMAs are highly correlated, these
coefficients must not be interpreted individually as causal feature
importance. The decisive evidence therefore comes from newly fitted ablation
models.

## 9. Ablation: What drives the Ridge signal?

Each variant was retrained using the same annual folds, test rows, and
evaluation rules.

| Feature set | Mean IC | Median IC | IC > 0 | Mean top-bottom |
| --- | ---: | ---: | ---: | ---: |
| Full Ridge | +0.1297 | +0.1452 | 71.2% | +42.64% |
| `volatility_60d` only | **+0.1324** | +0.1710 | 69.2% | **+43.03%** |
| Both volatility features only | +0.1317 | +0.1701 | 70.2% | +42.57% |
| Excluding return features | +0.1380 | +0.1764 | 73.1% | +43.25% |
| Excluding SMA features | +0.1382 | **+0.1872** | 73.1% | +42.86% |
| Excluding volatility features | +0.0148 | +0.0103 | 53.8% | +6.87% |
| Return features only | +0.0101 | +0.0301 | 57.7% | +2.75% |
| SMA distances only | -0.0159 | -0.0129 | 43.3% | -1.13% |

These results materially refine the research findings so far:

- The 60-day volatility feature alone reproduces the full Ridge model.
- Almost the entire rank-IC signal disappears without volatility.
- Returns and SMA distances carry little signal on their own. Adding either
  group to the volatility features increases mean IC slightly to about
  `+0.138`, while combining all nine features reduces it again to `+0.1297`.
  A stable incremental contribution has therefore not been established.
- Because the `volatility_60d` coefficient is positive in every fold, the
  single-feature ranking is effectively a simple high-to-low sort on 60-day
  volatility. No model fit is needed to produce this ordering.

The earlier working hypothesis that Ridge had discovered a broadly diversified
combination of technical signals is not supported by the ablation test.

## 10. Check using calendar-month cohorts

To reduce the heavy overlap among monthly 12-month targets, the Ridge results
were split into twelve start-month cohorts: all January observations, all
February observations, and so on. Successive observations within a cohort have
largely non-overlapping 12-month horizons.

- All twelve cohorts have a positive mean rank IC.
- Mean IC ranges from `+0.070` to `+0.179`.
- All twelve cohorts have a positive mean top-bottom spread, ranging from
  `+33.3%` to `+51.5%`.
- February is the weakest cohort, with a median IC of `-0.005` and only 44.4%
  positive annual observations.
- Each cohort contains only eight or nine observations.

The effect is therefore not solely the result of a favorable choice of calendar
month or repeated counting of nearly identical target periods. The small number
of independent years still precludes strong claims about statistical
significance.

## 11. Key methodological limitations

### Survivorship and selection bias

The universe was selected from the largest IVV positions in September 2026 and
applied retrospectively through 2017. The backtest therefore knows indirectly
which companies still exist and are large enough in 2026. Subsequent losers,
index deletions, acquisitions, bankruptcies, and delistings may be absent.

This bias is particularly serious for a positive high-volatility signal:
volatile winners can grow into today's large-cap universe, while volatile
losers disappear from it. The current test therefore cannot distinguish
between volatility having general predictive information and the later winners
having been selected in hindsight.

### Overlapping target periods

Monthly 252-session targets share most of their future price periods. The 104
IC months are therefore not 104 independent experiments. The twelve start-month
cohorts reduce this problem, but they do not resolve the small number of years
or common market-level and stock-level dependence.

### Missing controls for known risk factors

The analysis has not yet tested whether sector, size, liquidity, beta, or other
known factors explain the result. The volatility score must not be described as
standalone alpha without these controls.

### Data quality and implementability

The price audit flags four adjusted daily moves of at least +100% or at most
-50%: `TRGP`, `OXY`, `PCG`, and `MRNA`. These observations were not removed
automatically. The MRNA jump occurs in 2026 and is therefore outside the range
of fully observed outcomes to date. The remaining flags still need to be
reviewed as either genuine extreme events or potential data problems.

Delisting returns, transaction costs, turnover, slippage, and executable
rebalancing rules have not yet been modeled.

## 12. What is verified, unsupported, and unresolved

### Verified within the current sample

- A positive historical cross-sectional ranking signal exists.
- Ridge ranks better than the tested XGBoost configuration and 120-day
  momentum.
- Ranking quality and poor absolute calibration coexist.
- The probability models do not beat the constant baseline on Brier score.
- Nearly the entire Ridge signal is explained by `volatility_60d`.
- The signal remains positive on average in all twelve start-month cohorts.

### Unsupported or contradicted

- Ridge does not use a demonstrably valuable broad combination of the nine
  features.
- XGBoost provides no demonstrated incremental value over Ridge.
- The outputs are not reliable expected returns or outperformance
  probabilities.
- The 2018 and 2021 results do not yet identify distinct market regimes.

### Unresolved

- Does the complete research pipeline produce true null results with random
  labels and random features, and reliably detect known synthetic signals and
  intentional leakage?
- Does the volatility effect survive in a genuine point-in-time universe?
- Does it remain after neutralizing sector, size, liquidity, and beta?
- Is the effect statistically robust under block-based or year-based
  inference?
- Does an executable strategy remain after turnover and costs?
- Does the result replicate in an independent market or period?

## 13. Strongest defensible current claim

The strongest statement currently supported by the evidence is:

> Within a survivor universe constructed from today's large IVV holdings,
> stocks with higher trailing 60-day volatility had, on average, higher
> subsequent 12-month excess returns and a positive cross-sectional ranking
> from 2017 through August 2025. Whether this relationship exists outside the
> biased sample remains unresolved.

Statements that are not currently supported include:

- "StockResearch has demonstrated profitable alpha."
- "Ridge reliably predicts expected returns."
- "High volatility causes higher future returns."
- "The result will persist in the future."

## 14. Prioritized next experiments

### 0. Adversarial end-to-end validation of the research pipeline

Before interpreting any new market data, the measurement pipeline must be
tested on data with known ground truth to establish that it reliably
distinguishes signal from noise. These tests must pass through the same sample,
point-in-time fold, scoring or fitting, rank-IC, decile-assignment, and reporting
path as the real experiment. They must introduce neither new features nor
hyperparameter tuning.

The following four controls are mandatory:

1. **Random labels as a negative control.** Within each decision month, randomly
   permute the observed future excess returns among the stocks present at that
   time. Preserve universe size, cross-sectional distribution, target dates,
   and missingness. Then derive `future_outperform_12m` consistently from the
   permuted excess return rather than permuting it independently.
2. **Random features as a negative control.** Within each decision month,
   permute the frozen `volatility_60d` score independently of the outcomes.
   This preserves its monthly distribution, outliers, and missing values while
   breaking its assignment to individual stocks.
3. **A synthetic known signal as a positive control.** Create an artificial
   panel containing the same required schema fields and a predefined monotonic
   positive relationship between a feature available at the decision date and
   the subsequent outcome. A noise-free version must recover the correct
   ranking direction almost perfectly; a prespecified noisy version tests more
   realistic signal strengths.
4. **Intentional leakage as a positive sensitivity control.** In an isolated
   test path, expose a monotonic copy of the future outcome as a feature. The
   fitting, ranking, and reporting pipeline must respond with a rank IC near
   `+1` and an extreme decile spread. In a separate defensive test, the
   production path must explicitly reject target columns, `future_*` columns,
   and other fields that become known only after `as_of`. The first part shows
   sensitivity to leakage; only the second protects the real experiment from
   it.

A single random draw is not an adequate test. Both negative controls must use
at least 1,000 prespecified, numbered seeds. The full empirical null
distributions must be stored for mean and median rank IC, the fraction of
positive periods, and the top-bottom spread. The controls pass only if:

- the means of the null distributions show no systematic positive or negative
  offset; quantitatively, their distance from zero must be no greater than
  `0.1` standard deviations of the respective null distribution;
- the mean fraction of positive IC periods lies between 45% and 55%;
- the noise-free synthetic control produces mean and median rank IC of at least
  `+0.99`, a positive top-bottom spread, and the correct ranking direction;
- the leakage control also achieves at least `+0.99` mean rank IC, while the
  production path rejects the same leakage feature; and
- all seeds, inputs, interventions, metrics, and software versions are recorded
  in the run manifest or in hashed artifacts.

If any control fails, the point-in-time test must not be interpreted as evidence
for or against the market signal. The cause in the measurement or data path
must first be corrected, after which the unchanged control suite must be rerun.
Passing the red-team test establishes only that the research pipeline functions
internally. It does not validate historical membership data, corporate actions,
delisting returns, or the economic hypothesis itself.

### 1. Point-in-time falsification of the volatility signal

This is the next decisive test. It requires historical point-in-time membership
data and price histories that include index deletions, acquisitions, and
delistings. The raw `volatility_60d` score should be the primary baseline, while
Ridge remains frozen and unchanged. New features or hyperparameter tuning would
blur the research question.

The hypothesis and its technical implementation must be locked before viewing
the point-in-time results as follows:

- Inputs are adjusted closing prices and daily returns calculated from them
  using `pct_change(fill_method=None)`.
- `volatility_60d` is the unannualized sample standard deviation (`ddof=1`) of
  the last 60 daily returns, matching the existing `rolling(60).std()`
  implementation.
- Higher `volatility_60d` implies a better predicted rank. There is no
  winsorization, transformation, neutralization, or ex-post sign reversal.
- Decision frequency, the 252-XNYS-session target relative to SPY, tie handling,
  and minimum ranking breadth remain unchanged.
- The primary metrics remain mean and median rank IC, the fraction of positive
  IC periods, the top-bottom spread, and the same results by test year. A poor
  result must not be reinterpreted by changing the primary metric after the
  fact.

The current sample generation requires all nine features to be available,
including the 200-day SMA. For the frozen single-feature test, eligibility must
depend only on `volatility_60d`, the valid price at the decision date, and the
general data-quality rules. Otherwise, young stocks with 60 to 199 available
trading days would be excluded despite having a computable signal. This change
corrects the measurement universe; it is not feature tuning.

Membership for every decision date must be determined solely from universe
information available at that time. Subsequent index membership must not
retroactively determine inclusion or exclusion. Acquisitions, bankruptcies,
and delistings require either complete terminal returns or a prespecified
conservative treatment. A stock must not be silently removed from the IC or
decile return because a later price is unavailable. Because the current
evaluation marks an entire month as incomplete when one outcome is missing, the
extent and implications of missing delisting outcomes must be reported
explicitly before the run.

The point-in-time data must be evaluated only after this specification has been
versioned. Locking the specification does not retroactively turn the result
discovered in the survivor universe into a confirmatory finding. It does,
however, make the point-in-time test a genuine attempt to falsify this now-fixed
hypothesis.

### 2. Sector, size, liquidity, and beta neutralization

Scores and outcomes should be adjusted within comparable groups or through
cross-sectional residualization. This tests whether the result is merely a
hidden factor or sector exposure.

### 3. Time-dependent statistical inference

The evaluation should not treat individual monthly rows as independent.
Suitable approaches include year-level summaries, block bootstrap methods, and
prespecified start cohorts with largely non-overlapping periods.

### 4. Independent replication

The frozen volatility score should be tested without modification in another
equity universe, market, or previously untouched period.

### 5. Only then: portfolio and product evaluation

Turnover, transaction costs, rebalancing, capacity, and risk controls become
worth evaluating only if the signal survives the bias and factor tests. Until
then, neither expected return nor `P(Outperformance)` should be presented as a
reliable product metric.

## 15. Reproducibility and artifacts

The reference run is complete and its manifest contains artifact hashes. The
key local artifacts are located at:

- `data/runs/20260910T221307_4dbef5caf234/manifest.json`
- `data/runs/20260910T221307_4dbef5caf234/predictions/signal_report.md`
- `data/runs/20260910T221307_4dbef5caf234/predictions/model_comparison_summary.csv`
- `data/runs/20260910T221307_4dbef5caf234/predictions/model_comparison_by_year.csv`
- `data/runs/20260910T221307_4dbef5caf234/predictions/ranking_metrics_by_date.parquet`
- `data/runs/20260910T221307_4dbef5caf234/predictions/decile_returns_by_date.csv`
- `data/runs/20260910T221307_4dbef5caf234/predictions/price_jump_audit.csv`

The Ridge coefficient, ablation, and start-month analyses were reconstructed
read-only from this run's frozen samples and predictions. They have not yet been
implemented as separate, versioned run artifacts. For full machine-level
reproducibility, these analyses should later be incorporated into the pipeline
as fixed research reports.
