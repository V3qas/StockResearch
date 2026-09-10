import hashlib
import json

import pandas as pd
import pytest

from src import experiments
from src.experiments import ExperimentRun


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_successful_run_freezes_inputs_code_and_output_hashes(tmp_path):
    cache = tmp_path / "mutable_cache.parquet"
    original = pd.DataFrame({"Close": [100.0, 110.0]})
    original.to_parquet(cache)
    with ExperimentRun(tmp_path / "runs", {"end_date_exclusive": "2020-01-01"}) as run:
        pd.read_parquet(cache).to_parquet(run.raw / "stock_A.parquet")
        pd.DataFrame({"prediction": [0.1]}).to_csv(run.predictions / "summary.csv", index=False)
        assert not (run.runs_dir / "latest.json").exists()
        assert read_json(run.path / "manifest.json")["status"] == "running"
    pd.DataFrame({"Close": [999.0]}).to_parquet(cache)
    pd.testing.assert_frame_equal(pd.read_parquet(run.raw / "stock_A.parquet"), original)
    assert read_json(run.runs_dir / "latest.json")["run_id"] == run.run_id
    manifest = read_json(run.path / "manifest.json")
    assert manifest["status"] == "complete"
    assert "exchange-calendars" in manifest["versions"]
    assert "source/src/train.py" in manifest["artifacts_sha256"]
    for name, digest in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((run.path / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("failure", ["model", "artifact_write", "publish"])
def test_failed_run_never_changes_previous_successful_run(tmp_path, monkeypatch, failure):
    with ExperimentRun(tmp_path / "runs", {"name": "previous"}) as previous:
        pd.DataFrame({"prediction": [0.1]}).to_csv(previous.predictions / "summary.csv", index=False)
    old_pointer = (previous.runs_dir / "latest.json").read_bytes()
    old_artifacts = {p.relative_to(previous.path): p.read_bytes() for p in previous.path.rglob("*") if p.is_file()}
    replace = experiments.os.replace
    if failure == "publish":
        def fail_publish(source, destination):
            if destination.name == "latest.json":
                raise OSError("simulated publish failure")
            replace(source, destination)
        monkeypatch.setattr(experiments.os, "replace", fail_publish)
    with pytest.raises((RuntimeError, OSError)):
        with ExperimentRun(previous.runs_dir, {"name": "new"}) as failed:
            if failure == "model":
                raise RuntimeError("simulated training failure")
            if failure == "artifact_write":
                # Opening a directory as a file fails on every supported OS.
                (failed.predictions / "summary.csv").mkdir()
            pd.DataFrame({"prediction": [0.2]}).to_csv(failed.predictions / "summary.csv")
    assert (previous.runs_dir / "latest.json").read_bytes() == old_pointer
    assert read_json(failed.path / "manifest.json")["status"] == "failed"
    for name, content in old_artifacts.items():
        assert (previous.path / name).read_bytes() == content


def test_completed_empty_run_does_not_reuse_previous_results(tmp_path):
    with ExperimentRun(tmp_path / "runs", {}) as first:
        pd.DataFrame({"prediction": [0.1]}).to_parquet(first.predictions / "predictions.parquet")
    with ExperimentRun(first.runs_dir, {}) as empty:
        pd.DataFrame({"prediction": pd.Series(dtype=float)}).to_parquet(empty.predictions / "predictions.parquet")
    assert read_json(first.runs_dir / "latest.json")["run_id"] == empty.run_id
    assert pd.read_parquet(empty.predictions / "predictions.parquet").empty
    assert not pd.read_parquet(first.predictions / "predictions.parquet").empty
