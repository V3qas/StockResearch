from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research import ComparisonResult


def rank_deciles(predictions: pd.DataFrame, min_tickers: int = 10) -> pd.DataFrame:
    """Assign buckets from scores alone; ties share their average-rank bucket.

    Decile 1 is best. Midpoint percentiles put an exact multiple of ten distinct
    scores into equally sized buckets. Ties can make sizes unequal or empty.
    Unknown outcomes never change membership. Constant scores have no buckets.
    """
    columns = ["model", "ticker", "ranking_score", "alpha_12m"]
    frame = predictions[columns].copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if frame.reset_index(drop=True).assign(as_of=frame.index).duplicated(["model", "as_of", "ticker"]).any():
        raise ValueError("Duplicate model/as_of/ticker predictions.")
    frame["decile"] = np.nan
    pieces = []
    for _, group in frame.groupby(["model", frame.index], sort=True):
        score = group.ranking_score
        if len(group) >= max(10, min_tickers) and np.isfinite(score).all() and score.nunique() > 1:
            midpoint = (score.rank(ascending=False, method="average") - 0.5) / len(group)
            group["decile"] = np.floor(midpoint.to_numpy() * 10).astype(int) + 1
        pieces.append(group)
    return pd.concat(pieces) if pieces else frame


def decile_returns_by_date(assignments: pd.DataFrame) -> pd.DataFrame:
    """Publish returns only for complete dates, retaining all bucket coverage.

    Requiring every bucket to be populated gives each point in an aggregate
    curve the same dates. Partial dates keep membership/counts, but no returns.
    """
    rows = []
    for (model, as_of), group in assignments.groupby(["model", assignments.index], sort=True):
        complete = (
            group.decile.notna().all()
            and group.decile.nunique() == 10
            and np.isfinite(group.alpha_12m).all()
        )
        for decile in range(1, 11):
            bucket = group.loc[group.decile == decile]
            observed = int(np.isfinite(bucket.alpha_12m).sum())
            rows.append({
                "model": model, "as_of": as_of, "decile": decile,
                "n_tickers": len(bucket), "n_observed": observed,
                "label_coverage": observed / len(bucket) if len(bucket) else np.nan,
                "complete_date": bool(complete),
                "mean_excess_return": bucket.alpha_12m.mean() if complete else np.nan,
            })
    return pd.DataFrame(rows, columns=[
        "model", "as_of", "decile", "n_tickers", "n_observed", "label_coverage",
        "complete_date", "mean_excess_return",
    ])


def summarize_deciles(dated: pd.DataFrame) -> pd.DataFrame:
    return dated.groupby(["model", "decile"], sort=True).agg(
        mean_excess_return=("mean_excess_return", "mean"),
        evaluated_dates=("mean_excess_return", "count"),
        mean_bucket_size=("n_tickers", "mean"),
    )


def paired_rank_comparison(ranking: pd.DataFrame) -> pd.DataFrame:
    """Compare XGBoost to each baseline on the intersection of valid IC dates."""
    columns = ["baseline", "paired_dates", "xgboost_mean_rank_ic", "baseline_mean_rank_ic",
               "mean_difference", "positive_difference_fraction", "years_ahead", "paired_years"]
    if ranking.empty or "xgboost" not in set(ranking.model):
        return pd.DataFrame(columns=columns).set_index("baseline")
    table = ranking.reset_index().pivot(index=ranking.index.name or "index", columns="model", values="rank_ic")
    rows = []
    for baseline in table.columns.drop("xgboost"):
        pair = table[["xgboost", baseline]].dropna()
        delta = pair.xgboost - pair[baseline]
        annual = delta.groupby(delta.index.year).mean()
        rows.append({
            "baseline": baseline, "paired_dates": len(pair),
            "xgboost_mean_rank_ic": pair.xgboost.mean(), "baseline_mean_rank_ic": pair[baseline].mean(),
            "mean_difference": delta.mean(),
            "positive_difference_fraction": (delta > 0).mean() if len(delta) else np.nan,
            "years_ahead": int((annual > 0).sum()), "paired_years": len(annual),
        })
    return pd.DataFrame(rows, columns=columns).set_index("baseline")


def price_jump_audit(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flag large adjacent adjusted-close changes, without rewriting prices."""
    returns = prices.Close.pct_change(fill_method=None)
    flagged = returns.ge(1.0) | returns.le(-0.5)
    frame = pd.DataFrame({
        "ticker": ticker, "previous_close": prices.Close.shift(), "close": prices.Close,
        "daily_return": returns,
    }).loc[flagged]
    frame.index.name = "price_date"
    return frame


def _number(value, percent: bool = False) -> str:
    return "--" if pd.isna(value) else (f"{value:+.2%}" if percent else f"{value:+.4f}")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def _plot_diagnostics(comparison: ComparisonResult, deciles: pd.DataFrame, path: Path) -> bool:
    if comparison.ranking_by_date.rank_ic.notna().sum() == 0:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    models = comparison.summary.index[comparison.summary.ic_dates > 0]
    for model in models:
        annual = comparison.yearly_summary.xs(model, level="model")
        axes[0].plot(annual.index, annual.mean_rank_ic, marker="o", label=model)
        if not deciles.empty and model in deciles.index.get_level_values("model"):
            curve = deciles.xs(model, level="model")
            if curve.mean_excess_return.notna().any():
                axes[1].plot(curve.index, curve.mean_excess_return, marker="o", label=model)
    axes[0].set(title="Rank IC by prediction year", xlabel="Prediction year", ylabel="Mean Spearman rank IC")
    axes[1].set(title="Realized excess return by predicted decile", xlabel="Predicted decile (1 = highest score)", ylabel="Mean forward excess return")
    axes[1].set_xticks(range(1, 11))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle("StockResearch: exploratory signal diagnostics\nEqual-weight months; overlapping 252-session outcomes; partial years included", fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def write_signal_report(
    comparison: ComparisonResult,
    samples: pd.DataFrame,
    parameters: dict,
    output_dir: Path,
    *,
    price_jumps: pd.DataFrame | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = rank_deciles(comparison.predictions, parameters["min_ranking_tickers"])
    dated = decile_returns_by_date(assignments)
    deciles = summarize_deciles(dated)
    paired = paired_rank_comparison(comparison.ranking_by_date)
    assignments.to_parquet(output_dir / "rank_decile_assignments.parquet")
    dated.to_csv(output_dir / "decile_returns_by_date.csv", index=False)
    deciles.to_csv(output_dir / "decile_summary.csv")
    paired.to_csv(output_dir / "paired_rank_ic_comparison.csv")
    plotted = _plot_diagnostics(comparison, deciles, output_dir / "signal_diagnostics.png")
    if price_jumps is not None:
        price_jumps.to_csv(output_dir / "price_jump_audit.csv")

    def date_range(index) -> str:
        return "unavailable" if len(index) == 0 else f"{index.min():%Y-%m-%d} to {index.max():%Y-%m-%d}"

    pred = comparison.predictions
    observed = pred.loc[pred.alpha_12m.notna()]
    universe = parameters.get("universe", {})
    summary = comparison.summary
    counts = samples.groupby("ticker").size().reindex(parameters["tickers"], fill_value=0)
    coverage = pd.DataFrame({"eligible_samples": counts})
    coverage.index.name = "ticker"
    if "label_status" in samples:
        for status in ("observed", "pending", "missing"):
            coverage[status] = samples.loc[samples.label_status == status].groupby("ticker").size().reindex(coverage.index, fill_value=0)
    coverage.to_csv(output_dir / "universe_coverage.csv")
    no_samples = list(counts.index[counts == 0])
    monthly_counts = pred.loc[pred.model == summary.index[0]].groupby(level=0).ticker.nunique()
    eligible_range = "unavailable" if monthly_counts.empty else f"{monthly_counts.min()} to {monthly_counts.max()}"
    rows = []
    for model, row in summary.iterrows():
        rows.append([
            model, _number(row.mean_rank_ic), _number(row.median_rank_ic),
            "--" if pd.isna(row.positive_rank_ic_fraction) else f"{row.positive_rank_ic_fraction:.1%}",
            "--" if pd.isna(row.mae) else f"{row.mae * 100:.2f}", str(int(row.ic_dates)),
        ])
    text = [
        "# Does StockResearch find signal?", "",
        f"Universe: {len(parameters['tickers'])} requested stocks; {samples.ticker.nunique()} with eligible samples.",
        f"Eligible stocks per out-of-sample month: {eligible_range}. Without eligible features: {', '.join(no_samples) if no_samples else 'none'}.",
        f"Price request: {parameters['start_date']} to {parameters['end_date_exclusive']} (exclusive).",
        f"Forecast horizon: {parameters['forecast_days']} XNYS sessions (approximately 12 months); benchmark: {parameters['benchmark']} adjusted close.",
        f"Feature-eligible decision dates: {date_range(samples.index)}.",
        f"Out-of-sample prediction dates: {date_range(pred.index)}.",
        f"Decision dates with observed outcomes: {date_range(observed.index)}.", "",
        "**Exploratory experiment.** " + universe.get("limitations", "Fixed ticker selection; historical membership and delisting returns are not established."),
        "",
        *(["![Annual rank IC and realized decile returns](signal_diagnostics.png)", ""] if plotted else []),
        "## Model comparison", "",
        "IC below means cross-sectional **Spearman rank IC**. Months receive equal weight. "
        "IC>0 uses only defined IC months. MAE is in percentage points of excess return.", "",
        _table(["Model", "Mean rank IC", "Median rank IC", "IC>0", "MAE (pp)", "IC months"], rows), "",
        "Zero and the pooled historical mean cannot rank stocks: their scores are constant within a date. "
        "Their IC is undefined. Momentum is a 120-session ranking score; its MAE is undefined because it is not an alpha forecast.", "",
        "## XGBoost against baselines on shared IC months", "",
        _table(["Baseline", "Months", "XGBoost IC", "Baseline IC", "Difference", "Years XGBoost ahead"], [
            [name, str(int(row.paired_dates)), _number(row.xgboost_mean_rank_ic),
             _number(row.baseline_mean_rank_ic), _number(row.mean_difference),
             f"{int(row.years_ahead)}/{int(row.paired_years)}"]
            for name, row in paired.iterrows()
        ]), "",
    ]
    comparable = paired.loc[paired.paired_dates > 0]
    if "xgboost" not in summary.index or summary.loc["xgboost", "ic_dates"] == 0:
        text += ["No evaluable XGBoost ranking signal in this run.", ""]
    elif comparable.empty:
        text += ["No ranking baseline has shared evaluable months; incremental signal cannot be assessed.", ""]
    elif (comparable.mean_difference <= 0).any():
        names = ", ".join(comparable.index[comparable.mean_difference <= 0])
        text += [f"XGBoost does not beat {names} on mean rank IC over their shared months. "
                 "This run does not demonstrate an incremental advantage over all ranking baselines.", ""]
    else:
        text += ["XGBoost has a higher mean rank IC than the evaluable ranking baselines on shared months. "
                 "This is descriptive evidence only; universe selection, yearly stability and overlapping outcomes limit the conclusion.", ""]
    annual_rows = []
    if "xgboost" in comparison.yearly_summary.index.get_level_values("model"):
        for year, row in comparison.yearly_summary.xs("xgboost", level="model").iterrows():
            annual_rows.append([
                str(year), _number(row.mean_rank_ic), _number(row.median_rank_ic),
                str(int(row.ic_dates)), str(int(row.observed_labels)),
                str(int(row.get("pending_labels", 0))), str(int(row.get("missing_labels", 0))),
            ])
    text += [
        "## XGBoost by prediction year", "",
        _table(["Year", "Mean rank IC", "Median rank IC", "IC months", "Observed labels", "Pending", "Missing"], annual_rows), "",
        "Partial years are shown with their actual coverage. Future 12-month outcomes are not inferred.", "",
        "## Realized excess return by predicted decile", "",
        "Decile 1 is the highest predicted score. Stocks receive equal weight within each month; "
        "bucket means receive equal weight across complete months. Average-rank ties stay together. "
        f"A date requires at least {max(10, parameters['min_ranking_tickers'])} eligible stocks, all outcomes/scores and ten populated buckets. "
        "Assignments are saved even when outcomes are unknown. The spread is a forward-return diagnostic, not a tradable portfolio return.", "",
    ]
    for model in summary.index:
        if deciles.empty or model not in deciles.index.get_level_values("model"):
            continue
        curve = deciles.xs(model, level="model")
        if curve.evaluated_dates.sum() == 0:
            continue
        text += [f"### {model}", "", _table(["Predicted rank", "Realized 12M excess return", "Months"], [
            ["Top 10%" if bucket == 1 else "Bottom 10%" if bucket == 10 else f"{(bucket-1)*10}-{bucket*10}%",
             _number(row.mean_excess_return, True), str(int(row.evaluated_dates))]
            for bucket, row in curve.iterrows()
        ]), "", f"Top minus bottom: {_number(curve.loc[1, 'mean_excess_return'] - curve.loc[10, 'mean_excess_return'], True)}.", ""]
    if dated.empty or not dated.complete_date.any():
        text += ["No complete, populated decile dates are available.", ""]
    audit_count = len(price_jumps) if price_jumps is not None else None
    text += [
        "## Method and limits", "",
        f"- Expanding annual fits; {parameters['min_train_years']} prior calendar years required. "
        "Training labels must mature no later than the first prediction date of the fold. Scaling uses training data only.",
        "- Fixed existing model parameters; no tuning on this evaluation period.",
        "- Monthly 252-session outcomes overlap. IC months and yearly means are dependent; no significance or confidence claim is made.",
        "- Complete-date IC and deciles can select a subset of months when outcomes are missing; the dated artifacts retain coverage.",
        "- MAE uses observed target/forecast pairs. Delisting outcomes are not imputed. No transaction costs or executable rebalancing simulation.",
        f"- Adjacent adjusted-close jumps >= +100% or <= -50%: {audit_count if audit_count is not None else 'not audited'}. "
        "These are review flags, not confirmed data errors; prices are not corrected or excluded automatically.",
        "- Universe metadata, exact input snapshots, source, package versions and artifact hashes are stored with the run.",
    ]
    if audit_count:
        text += ["", "**Data review required:** flagged price jumps may distort features or outcomes. Inspect price_jump_audit.csv before interpreting model quality."]
    if universe.get("source_url"):
        text += ["", f"Universe source: [iShares holdings]({universe['source_url']}), as of {universe.get('holdings_as_of', 'unknown')}."]
    path = output_dir / "signal_report.md"
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path
