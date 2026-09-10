import json

import pandas as pd
import pytest

import app


def test_latest_as_of_rows_uses_decision_index_not_snapshot_date():
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "snapshot_date": pd.to_datetime(["2024-03-01"] * 3),
        },
        index=pd.DatetimeIndex(
            ["2024-01-31", "2024-02-29", "2024-02-29"],
            name="as_of",
        ),
    )

    latest = app.latest_as_of_rows(frame)

    assert latest.ticker.tolist() == ["A", "B"]
    assert latest.as_of.eq(pd.Timestamp("2024-02-29")).all()
    assert latest.index.name is None
    assert latest.sort_values("as_of").as_of.is_monotonic_increasing


def test_load_run_reloads_latest_pointer(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(app, "RUNS_DIR", runs_dir)

    for run_id in ("first", "second"):
        run_path = runs_dir / run_id
        run_path.mkdir()
        (run_path / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "status": "complete"}),
            encoding="utf-8",
        )

    (runs_dir / "latest.json").write_text(json.dumps({"run_id": "first"}), encoding="utf-8")
    first_path, _ = app.load_run()
    (runs_dir / "latest.json").write_text(json.dumps({"run_id": "second"}), encoding="utf-8")
    second_path, _ = app.load_run()

    assert first_path.name == "first"
    assert second_path.name == "second"


def test_load_run_rejects_path_traversal(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(app, "RUNS_DIR", runs_dir)
    (runs_dir / "latest.json").write_text(json.dumps({"run_id": "../outside"}), encoding="utf-8")

    with pytest.raises(ValueError, match="ungültige run_id"):
        app.load_run()
