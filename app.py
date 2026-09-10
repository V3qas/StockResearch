from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


st.set_page_config(
    page_title="StockResearch | Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --ink: #172033; --muted: #64748b; }
        .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1440px; }
        [data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #64748b; }
        h1 { letter-spacing: -0.04em; color: var(--ink); }
        h2, h3 { letter-spacing: -0.02em; color: var(--ink); }
        .eyebrow { color: #2563eb; font-size: .78rem; font-weight: 700;
                   letter-spacing: .12em; text-transform: uppercase; margin-bottom: .25rem; }
        .hero-copy { color: var(--muted); font-size: 1.05rem; margin-top: -.5rem; }
        .status-pill { display: inline-block; border-radius: 999px; padding: .28rem .65rem;
                       background: #dcfce7; color: #166534; font-weight: 700; font-size: .8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_pct(value: object) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def format_number(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def load_run() -> tuple[Path, dict]:
    """Resolve the latest pointer on every rerun so newly published runs appear."""
    pointer_path = RUNS_DIR / "latest.json"
    if not pointer_path.exists():
        raise FileNotFoundError("Kein Lauf gefunden: data/runs/latest.json fehlt.")

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run_id = pointer.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("latest.json enthält keine run_id.")

    runs_dir = RUNS_DIR.resolve()
    run_path = (runs_dir / run_id).resolve()
    if run_path.parent != runs_dir:
        raise ValueError("latest.json enthält eine ungültige run_id.")
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest des Laufs {run_id} fehlt.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Der letzte Lauf ist nicht als erfolgreich markiert.")
    if manifest.get("run_id") != run_id:
        raise ValueError("run_id in latest.json und Manifest stimmen nicht überein.")
    return run_path, manifest


@st.cache_data(show_spinner=False)
def read_artifacts(run_path: Path) -> dict[str, pd.DataFrame | None]:
    predictions_dir = run_path / "predictions"
    processed_dir = run_path / "processed"

    def read_parquet(path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None

    def read_csv(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    summary = read_csv(predictions_dir / "model_comparison_summary.csv")
    predictions = read_parquet(predictions_dir / "model_comparison_predictions.parquet")
    samples = read_parquet(processed_dir / "all_samples.parquet")
    by_year = read_csv(predictions_dir / "model_comparison_by_year.csv")
    ranking = read_parquet(predictions_dir / "ranking_metrics_by_date.parquet")
    coverage = read_csv(processed_dir / "benchmark_coverage.csv")

    date_columns = (
        "as_of", "session", "price_date", "future_target_date",
        "future_price_date", "benchmark_price_date",
        "benchmark_future_price_date", "snapshot_date", "date",
    )
    for frame in (predictions, samples, ranking, coverage):
        if frame is not None:
            for column in date_columns:
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")

    return {
        "summary": summary,
        "predictions": predictions,
        "samples": samples,
        "by_year": by_year,
        "ranking": ranking,
        "coverage": coverage,
    }


def with_as_of_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the decision date consistently whether parquet stored it as an index or column."""
    result = frame.copy()
    if "as_of" in result.columns:
        result["as_of"] = pd.to_datetime(result["as_of"], errors="coerce")
    elif isinstance(result.index, pd.DatetimeIndex) or result.index.name == "as_of":
        dates = pd.to_datetime(result.index, errors="coerce").to_numpy()
        result = result.reset_index(drop=True)
        result.insert(0, "as_of", dates)
    return result


def latest_as_of_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows for the most recent valid decision date."""
    result = with_as_of_column(frame)
    if "as_of" not in result.columns:
        return result.iloc[0:0]
    latest = result["as_of"].max()
    if pd.isna(latest):
        return result.iloc[0:0]
    return result.loc[result["as_of"].eq(latest)]


def show_error(error: Exception) -> None:
    st.error(str(error))
    st.info(
        "Starte zuerst einen vollständigen Research-Lauf mit `main.py`. "
        "Danach liest die GUI den Lauf über `data/runs/latest.json` ein."
    )


def sidebar(manifest: dict, artifacts: dict[str, pd.DataFrame | None]) -> tuple[str | None, str | None]:
    parameters = manifest.get("parameters", {})
    summary = artifacts["summary"]
    predictions = artifacts["predictions"]
    samples = artifacts["samples"]

    with st.sidebar:
        st.markdown("## 📈 StockResearch")
        st.caption("Lokales Research-Dashboard")
        st.divider()
        st.markdown("**Letzter Lauf**")
        st.markdown('<span class="status-pill">● erfolgreich</span>', unsafe_allow_html=True)
        st.caption(manifest.get("run_id", "unbekannt"))
        st.caption(f"Snapshot: {parameters.get('end_date_exclusive', '—')}")
        st.divider()

        model: str | None = None
        ticker: str | None = None
        if predictions is not None and "model" in predictions.columns:
            models = sorted(predictions["model"].dropna().astype(str).unique())
            if models:
                default_model = models.index("xgboost") if "xgboost" in models else 0
                model = st.selectbox("Modell", models, index=default_model)
        if samples is not None and "ticker" in samples.columns:
            tickers = sorted(samples["ticker"].dropna().astype(str).unique())
            if tickers:
                ticker = st.selectbox("Ticker", tickers, index=0)

        st.divider()
        st.caption("Datenquellen")
        st.caption(
            f"{len(summary) if summary is not None else 0} Modelle · "
            f"{len(parameters.get('tickers', []))} Ticker · "
            f"Horizont {parameters.get('forecast_days', '—')} Tage"
        )
    return model, ticker


def show_overview(manifest: dict, artifacts: dict[str, pd.DataFrame | None]) -> None:
    summary = artifacts["summary"]
    predictions = artifacts["predictions"]
    samples = artifacts["samples"]
    parameters = manifest.get("parameters", {})

    st.markdown('<div class="eyebrow">Research overview</div>', unsafe_allow_html=True)
    st.title("Research Dashboard")
    st.markdown(
        '<div class="hero-copy">Modellqualität, Markt-Ranking und Ticker-Signale auf einen Blick.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    latest_date = "—"
    if samples is not None and "price_date" in samples.columns:
        latest = pd.to_datetime(samples["price_date"], errors="coerce").max()
        latest_date = latest.strftime("%d.%m.%Y") if pd.notna(latest) else "—"
    coverage = None
    if summary is not None and "label_coverage" in summary.columns:
        coverage = summary["label_coverage"].dropna().max()

    metrics = st.columns(4)
    metrics[0].metric("Modelle", len(summary) if summary is not None else 0)
    metrics[1].metric("Ticker", len(parameters.get("tickers", [])))
    metrics[2].metric("Letzter Kursstand", latest_date)
    metrics[3].metric("Label-Abdeckung", format_pct(coverage))

    if summary is None or summary.empty:
        st.warning("Für diesen Lauf ist kein Modellvergleich vorhanden.")
        return

    st.subheader("Modellvergleich")
    overview = summary.copy()
    display_columns = [
        "model", "mae", "directional_accuracy", "brier_score", "mean_rank_ic", "observed_labels"
    ]
    display_columns = [column for column in display_columns if column in overview.columns]
    overview = overview[display_columns].rename(
        columns={
            "model": "Modell",
            "mae": "MAE",
            "directional_accuracy": "Trefferquote",
            "brier_score": "Brier Score",
            "mean_rank_ic": "Ø Rank-IC",
            "observed_labels": "Labels",
        }
    )
    st.dataframe(
        overview.style.format(
            {"MAE": "{:.3f}", "Trefferquote": "{:.1%}", "Brier Score": "{:.4f}", "Ø Rank-IC": "{:.4f}"},
            na_rep="—",
        ),
        width="stretch",
        hide_index=True,
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Brier Score — niedriger ist besser")
        if {"model", "brier_score"}.issubset(summary.columns):
            chart = summary[["model", "brier_score"]].dropna().set_index("model")
            if not chart.empty:
                st.bar_chart(chart, width="stretch")
    with chart_cols[1]:
        st.caption("Mittlerer Rank-IC — höher ist besser")
        if {"model", "mean_rank_ic"}.issubset(summary.columns):
            chart = summary[["model", "mean_rank_ic"]].dropna().set_index("model")
            if not chart.empty:
                st.bar_chart(chart, width="stretch")

    if predictions is not None and not predictions.empty:
        st.subheader("Letzte Modell-Signale")
        latest_predictions = latest_as_of_rows(predictions)
        signal_columns = [
            "as_of", "ticker", "model", "predicted_probability_outperform_12m", "predicted_alpha_12m",
            "predicted_rank", "label_status",
        ]
        signal_columns = [column for column in signal_columns if column in latest_predictions.columns]
        if signal_columns and not latest_predictions.empty:
            sort_columns = [column for column in ("predicted_rank", "ticker", "model") if column in latest_predictions]
            signal_view = latest_predictions[signal_columns]
            if sort_columns:
                signal_view = signal_view.sort_values(sort_columns)
            st.dataframe(
                signal_view,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Keine datierten Modell-Signale vorhanden.")


def show_model_comparison(artifacts: dict[str, pd.DataFrame | None]) -> None:
    summary = artifacts["summary"]
    by_year = artifacts["by_year"]
    if summary is None or summary.empty:
        st.warning("Keine Modellvergleichsdaten vorhanden.")
        return

    st.header("Modellvergleich")
    st.caption("Die Kennzahlen stammen aus dem gespeicherten Out-of-sample-Backtest.")
    st.dataframe(summary, width="stretch", hide_index=True)

    if by_year is not None and not by_year.empty:
        st.subheader("Entwicklung nach Jahr")
        models = sorted(by_year["model"].dropna().astype(str).unique()) if "model" in by_year else []
        year_model = (
            st.selectbox(
                "Modell für Jahresansicht",
                models,
                index=models.index("xgboost") if "xgboost" in models else 0,
            )
            if models else None
        )
        yearly = by_year.loc[by_year["model"].eq(year_model)].copy() if year_model else by_year.copy()
        if "test_year" in yearly.columns:
            yearly = yearly.set_index("test_year")
        plot_columns = [
            column for column in ("brier_score", "directional_accuracy", "mean_rank_ic")
            if column in yearly.columns
        ]
        if plot_columns:
            st.line_chart(yearly[plot_columns], width="stretch")


def show_ticker_analysis(artifacts: dict[str, pd.DataFrame | None], ticker: str | None, model: str | None) -> None:
    samples = artifacts["samples"]
    predictions = artifacts["predictions"]
    if samples is None or samples.empty or ticker is None:
        st.warning("Keine Ticker-Daten vorhanden.")
        return

    st.header(f"Ticker-Analyse · {ticker}")
    ticker_samples = samples.loc[samples["ticker"].astype(str).eq(ticker)].copy()
    if "price_date" in ticker_samples.columns:
        ticker_samples = ticker_samples.sort_values("price_date")
    if ticker_samples.empty:
        st.warning("Für diesen Ticker wurden keine Daten gefunden.")
        return

    latest = ticker_samples.iloc[-1]
    stats = st.columns(4)
    stats[0].metric("Letzter Schlusskurs", format_number(latest.get("Close")))
    stats[1].metric("Rendite 20 Tage", format_pct(latest.get("return_20d")))
    stats[2].metric("Rendite 120 Tage", format_pct(latest.get("return_120d")))
    stats[3].metric("Volatilität 20 Tage", format_pct(latest.get("volatility_20d")))

    if "price_date" in ticker_samples.columns and "Close" in ticker_samples.columns:
        st.subheader("Kursverlauf")
        prices = ticker_samples.set_index("price_date")["Close"].dropna().tail(260)
        st.line_chart(prices, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Aktuelle Merkmale")
        feature_columns = [
            "price_date", "Close", "return_5d", "return_20d", "return_60d", "return_120d",
            "distance_sma_20", "distance_sma_50", "distance_sma_200", "label_status",
        ]
        feature_columns = [column for column in feature_columns if column in ticker_samples.columns]
        st.dataframe(ticker_samples[feature_columns].tail(12), width="stretch", hide_index=True)
    with right:
        st.subheader("Modell-Signal")
        if predictions is None or predictions.empty:
            st.info("Keine Vorhersagen vorhanden.")
        else:
            ticker_predictions = predictions.loc[predictions["ticker"].astype(str).eq(ticker)].copy()
            if model and "model" in ticker_predictions.columns:
                ticker_predictions = ticker_predictions.loc[ticker_predictions["model"].eq(model)]
            ticker_predictions = latest_as_of_rows(ticker_predictions)
            signal_columns = [
                "as_of", "model", "predicted_probability_outperform_12m", "predicted_alpha_12m",
                "predicted_rank", "label_status",
            ]
            signal_columns = [column for column in signal_columns if column in ticker_predictions.columns]
            if ticker_predictions.empty:
                st.info("Kein datiertes Signal für diese Auswahl vorhanden.")
            else:
                st.dataframe(ticker_predictions[signal_columns], width="stretch", hide_index=True)


def show_ranking(artifacts: dict[str, pd.DataFrame | None], model: str | None) -> None:
    ranking = artifacts["ranking"]
    if ranking is None or ranking.empty:
        st.info("Für diesen Lauf sind keine Ranking-Metriken gespeichert.")
        return

    st.header("Ranking-Qualität")
    st.caption("IC und Rank-IC messen, wie gut das Modell die relative Reihenfolge der Ticker trifft.")
    ranking = ranking.copy()
    if model and "model" in ranking.columns:
        ranking = ranking.loc[ranking["model"].eq(model)]
    ranking = with_as_of_column(ranking)
    if "as_of" in ranking.columns:
        ranking = ranking.sort_values("as_of").set_index("as_of")
    elif "date" in ranking.columns:
        ranking = ranking.sort_values("date").set_index("date")
    plot_columns = [column for column in ("ic", "rank_ic") if column in ranking.columns]
    if plot_columns:
        st.line_chart(ranking[plot_columns], width="stretch")
    st.dataframe(ranking.reset_index().tail(30), width="stretch", hide_index=True)


def show_data_quality(manifest: dict, artifacts: dict[str, pd.DataFrame | None]) -> None:
    st.header("Datenqualität & Laufdetails")
    parameters = manifest.get("parameters", {})
    left, right = st.columns(2)
    with left:
        st.subheader("Parameter")
        parameter_rows = {key: value for key, value in parameters.items() if key not in {"feature_columns", "models"}}
        parameter_rows = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
            for key, value in parameter_rows.items()
        }
        st.dataframe(
            pd.DataFrame({"Parameter": parameter_rows.keys(), "Wert": parameter_rows.values()}),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Artefakte")
        artifact_rows = pd.DataFrame(
            {
                "Artefakt": list(artifacts.keys()),
                "Status": ["vorhanden" if frame is not None else "fehlt" for frame in artifacts.values()],
                "Zeilen": [len(frame) if frame is not None else 0 for frame in artifacts.values()],
            }
        )
        st.dataframe(artifact_rows, hide_index=True, width="stretch")

    coverage = artifacts["coverage"]
    if coverage is not None and not coverage.empty:
        st.subheader("Benchmark-Abdeckung")
        st.dataframe(coverage, width="stretch", hide_index=True)

    with st.expander("Manifest anzeigen"):
        st.json(manifest)


def main() -> None:
    inject_styles()
    try:
        run_path, manifest = load_run()
        artifacts = read_artifacts(run_path)
    except (FileNotFoundError, KeyError, OSError, ValueError, pd.errors.EmptyDataError) as error:
        show_error(error)
        st.stop()

    model, ticker = sidebar(manifest, artifacts)
    tabs = st.tabs(["Übersicht", "Modellvergleich", "Ticker-Analyse", "Ranking-Qualität", "Datenqualität"])
    with tabs[0]:
        show_overview(manifest, artifacts)
    with tabs[1]:
        show_model_comparison(artifacts)
    with tabs[2]:
        show_ticker_analysis(artifacts, ticker, model)
    with tabs[3]:
        show_ranking(artifacts, model)
    with tabs[4]:
        show_data_quality(manifest, artifacts)

    st.caption(f"Quelle: {run_path.relative_to(PROJECT_ROOT)} · Schema v{manifest.get('schema_version', '—')}")


if __name__ == "__main__":
    main()
