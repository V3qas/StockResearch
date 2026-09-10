"""Re-fit the recorded comparison from verified, frozen inputs without downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from src.config import RUNS_DIR
from src.experiments import ExperimentRun
from src.research import default_model_specs, run_model_comparison
from src.signal_report import price_jump_audit, write_signal_report


def verify_inputs(source: Path) -> dict:
    source = source.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Replay requires a completed run.")
    artifacts = manifest.get("artifacts_sha256", {})
    required = {"processed/all_samples.parquet", "raw/benchmark.parquet", "raw/reference_calendar.parquet"}
    inputs = {name for name in artifacts if name.startswith(("raw/", "processed/"))}
    if not required.issubset(inputs):
        raise ValueError("Run lacks hashed frozen inputs.")
    for name in sorted(inputs):
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Invalid frozen input path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifacts[name]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    return manifest


def replay(source: Path) -> Path:
    source = source.resolve()
    manifest = verify_inputs(source)
    parameters = dict(manifest["parameters"])
    if not parameters.get("comparison_requested"):
        raise ValueError("Source run must contain a model comparison.")
    by_name = {spec.name: spec for spec in default_model_specs()}
    specs = [by_name[name] for name in parameters["models"]]
    parameters["replay_of"] = manifest["run_id"]
    parameters["replay_source_manifest_sha256"] = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    with ExperimentRun(RUNS_DIR, parameters) as run:
        for name in manifest["artifacts_sha256"]:
            if name.startswith(("raw/", "processed/")):
                target = run.path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
        (run.raw / "replay_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        samples = pd.read_parquet(run.processed / "all_samples.parquet")
        comparison = run_model_comparison(
            samples, min_train_years=parameters["min_train_years"],
            min_train_samples=parameters["min_train_samples"],
            min_ranking_tickers=parameters["min_ranking_tickers"], model_specs=specs,
            feature_columns=parameters["feature_columns"],
        )
        predictions = comparison.predictions
        predictions.loc[predictions.model == "xgboost"].to_parquet(run.predictions / "cross_sectional_backtest_predictions.parquet")
        predictions.to_parquet(run.predictions / "model_comparison_predictions.parquet")
        comparison.summary.to_csv(run.predictions / "model_comparison_summary.csv")
        comparison.yearly_summary.to_csv(run.predictions / "model_comparison_by_year.csv")
        comparison.ranking_by_date.to_parquet(run.predictions / "ranking_metrics_by_date.parquet")
        comparison.classification_comparison.to_csv(run.predictions / "classification_common_cohort.csv")
        jumps = [price_jump_audit(pd.read_parquet(run.raw / "benchmark.parquet"), parameters["benchmark"])]
        from src.data_loader import ticker_to_filename
        for ticker in parameters["tickers"]:
            jumps.append(price_jump_audit(pd.read_parquet(run.raw / f"stock_{ticker_to_filename(ticker)}.parquet"), ticker))
        write_signal_report(comparison, samples, parameters, run.predictions, price_jumps=pd.concat(jumps))
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed data/runs/<run_id> directory.")
    args = parser.parse_args()
    print(f"Completed replay: {replay(args.run)}")
